import cv2
import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

CONFIG_RODOVIAS = {
    "arao_sahm_km95": {
        "nome": "Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP)",
        "camera_url": "https://34.104.32.249.nip.io/SP008-KM095/stream.m3u8",
        "line_start_ratio": (0.18, 0.33),
        "line_end_ratio": (0.81, 0.53),
        "band_half_thickness": 120,
    },
    "manoel_hyppolito_km83": {
        "nome": "Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP)",
        "camera_url": "https://34.104.32.249.nip.io/SP055-KM083/stream.m3u8",
        "line_start_ratio": (0.10, 0.55),
        "line_end_ratio": (0.90, 0.55),
        "band_half_thickness": 110,
    },
    "floriano_km26": {
        "nome": "Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP)",
        "camera_url": "https://34.104.32.249.nip.io/SP123-KM026A/stream.m3u8",
        "line_start_ratio": (0.10, 0.55),
        "line_end_ratio": (0.90, 0.55),
        "band_half_thickness": 110,
    },
}

MIN_AREA = 1200
THRESHOLD_VALUE = 25
RESET_FRAMES = 10
MIN_WIDTH = 35
MIN_HEIGHT = 25
BLUR_SIZE = (5, 5)

TARGET_X_DEFAULT = 8
MARGIN_DEFAULT = 3.0

PROCESS_SCALE = 0.60
MJPEG_SLEEP = 0.08


class EstadoContagem:
    def __init__(self, nome_rodovia):
        self.nome_rodovia = nome_rodovia
        self.lock = threading.Lock()

        self.raw_frame = None
        self.frame_id = 0

        self.count = 0
        self.events = deque()
        self.last_frame = None
        self.running = False
        self.last_update = None
        self.gate_occupied = False
        self.empty_frames = 0
        self.prev_gray = None
        self.target_x = TARGET_X_DEFAULT
        self.margin = MARGIN_DEFAULT
        self.last_signal = "SEM_DADOS"
        self.last_trend = "SEM_DADOS"

    def set_raw_frame(self, frame):
        with self.lock:
            self.raw_frame = frame
            self.frame_id += 1

    def get_raw_frame(self):
        with self.lock:
            if self.raw_frame is None:
                return None, self.frame_id
            return self.raw_frame.copy(), self.frame_id

    def set_frame(self, frame):
        with self.lock:
            self.last_frame = frame
            self.last_update = time.time()

    def set_params(self, target_x=None, margin=None):
        with self.lock:
            if target_x is not None:
                self.target_x = target_x
            if margin is not None:
                self.margin = margin

    def get_snapshot(self):
        with self.lock:
            now = time.time()

            while self.events and now - self.events[0] > 15 * 60:
                self.events.popleft()

            total5 = sum(1 for e in self.events if now - e <= 5 * 60)
            total10 = sum(1 for e in self.events if now - e <= 10 * 60)
            total15 = len(self.events)

            rate5 = total5 / 5
            rate10 = total10 / 10
            rate15 = total15 / 15

            flow_index_510 = rate5 / rate10 if rate10 > 0 else 0.0
            flow_index_515 = rate5 / rate15 if rate15 > 0 else 0.0

            if rate15 == 0 and rate5 == 0:
                trend = "SEM_DADOS"
            elif flow_index_515 >= 1.15:
                trend = "ACELERANDO"
            elif flow_index_515 <= 0.85:
                trend = "CAINDO"
            else:
                trend = "ESTAVEL"

            adjustment = 1.0
            if flow_index_515 > 1.15:
                adjustment = 1.2
            elif flow_index_515 < 0.85:
                adjustment = 0.8

            forecast_rate = (rate5 * 0.6) + (rate10 * 0.3) + (rate15 * 0.1)
            forecast_rate_adjusted = forecast_rate * adjustment
            forecast_5m = forecast_rate_adjusted * 5

            diff = forecast_5m - self.target_x
            if abs(diff) < self.margin:
                signal = "NAO_APOSTAR"
            elif diff >= self.margin:
                signal = "OVER"
            elif diff <= -self.margin:
                signal = "UNDER"
            else:
                signal = "NAO_APOSTAR"

            self.last_signal = signal
            self.last_trend = trend

            return {
                "nome_rodovia": self.nome_rodovia,
                "running": self.running,
                "count_total": self.count,
                "target_x": self.target_x,
                "margin": self.margin,
                "veiculos_5min": total5,
                "veiculos_10min": total10,
                "veiculos_15min": total15,
                "rate5": round(rate5, 4),
                "rate10": round(rate10, 4),
                "rate15": round(rate15, 4),
                "flow_index_510": round(flow_index_510, 4),
                "flow_index_515": round(flow_index_515, 4),
                "forecast_5m": round(forecast_5m, 4),
                "signal": signal,
                "trend": trend,
                "last_update": self.last_update
            }


estados = {
    key: EstadoContagem(config["nome"])
    for key, config in CONFIG_RODOVIAS.items()
}

app = FastAPI(title="Monitor Fluxo Rodovias")


def obter_estado(key: str) -> EstadoContagem:
    estado = estados.get(key)
    if estado is None:
        raise HTTPException(status_code=404, detail=f"Rodovia '{key}' não encontrada.")
    return estado


def open_capture(camera_url):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


def ratio_to_point(ratio_point, width, height):
    x = int(width * ratio_point[0])
    y = int(height * ratio_point[1])
    return (x, y)


def build_line_band(line_start, line_end, half_thickness):
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        return np.array([[x1, y1], [x1, y1], [x1, y1], [x1, y1]], dtype=np.int32)

    px = -dy / length
    py = dx / length

    p1 = (int(x1 + px * half_thickness), int(y1 + py * half_thickness))
    p2 = (int(x2 + px * half_thickness), int(y2 + py * half_thickness))
    p3 = (int(x2 - px * half_thickness), int(y2 - py * half_thickness))
    p4 = (int(x1 - px * half_thickness), int(y1 - py * half_thickness))

    return np.array([p1, p2, p3, p4], dtype=np.int32)


def point_to_line_distance(px, py, x1, y1, x2, y2):
    line_mag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if line_mag == 0:
        return 999999
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_mag


def rect_intersects_line(rect, line_start, line_end, tolerance=18):
    x, y, w, h = rect
    x1, y1 = line_start
    x2, y2 = line_end

    pontos = [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h),
        (x + w // 2, y + h // 2),
    ]

    for px, py in pontos:
        dist = point_to_line_distance(px, py, x1, y1, x2, y2)
        if dist <= tolerance:
            return True

    return False


def rect_intersects_band(rect, band_polygon):
    x, y, w, h = rect

    pontos = [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h),
        (x + w // 2, y + h // 2),
    ]

    for px, py in pontos:
        inside = cv2.pointPolygonTest(band_polygon, (float(px), float(py)), False)
        if inside >= 0:
            return True

    return False


def draw_overlay(frame, nome_rodovia, band_polygon, line_start, line_end, snapshot, best_rect=None, best_center=None):
    out = frame.copy()

    overlay = out.copy()
    cv2.fillPoly(overlay, [band_polygon], color=(0, 255, 255))
    out = cv2.addWeighted(overlay, 0.10, out, 0.90, 0)
    cv2.polylines(out, [band_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.line(out, line_start, line_end, (0, 255, 0), 4)

    if best_rect is not None:
        x, y, w, h = best_rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if best_center is not None:
        cx, cy = best_center
        cv2.circle(out, (cx, cy), 10, (0, 255, 255), -1)
        cv2.putText(out, f"x:{cx} y:{cy}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    linhas = [
        nome_rodovia,
        f"Contagem: {snapshot['count_total']}",
        f"5m: {snapshot['veiculos_5min']} | 10m: {snapshot['veiculos_10min']} | 15m: {snapshot['veiculos_15min']}",
        f"Rate5: {snapshot['rate5']:.2f} | Rate10: {snapshot['rate10']:.2f} | Rate15: {snapshot['rate15']:.2f}",
        f"Prev 5m: {snapshot['forecast_5m']:.2f} | Sinal: {snapshot['signal']} | Tendencia: {snapshot['trend']}",
        f"Meta X: {snapshot['target_x']} | Margem: {snapshot['margin']}"
    ]

    y0 = 30
    for i, texto in enumerate(linhas):
        cv2.putText(out, texto, (20, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    return out


def capturar_camera(key, config):
    estado = estados[key]
    camera_url = config["camera_url"]

    while True:
        cap = open_capture(camera_url)

        if not cap.isOpened():
            with estado.lock:
                estado.running = False
            time.sleep(3)
            continue

        with estado.lock:
            estado.running = True

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                with estado.lock:
                    estado.running = False
                break

            estado.set_raw_frame(frame)
            time.sleep(0.001)

        cap.release()
        time.sleep(2)


def processar_camera(key, config):
    estado = estados[key]
    band_half_thickness = int(config.get("band_half_thickness", 110))
    ultimo_frame_id_processado = -1

    with estado.lock:
        estado.prev_gray = None
        estado.gate_occupied = False
        estado.empty_frames = 0

    while True:
        frame, frame_id = estado.get_raw_frame()

        if frame is None or frame_id == ultimo_frame_id_processado:
            time.sleep(0.01)
            continue

        ultimo_frame_id_processado = frame_id

        h, w = frame.shape[:2]

        line_start = ratio_to_point(config["line_start_ratio"], w, h)
        line_end = ratio_to_point(config["line_end_ratio"], w, h)
        band_polygon = build_line_band(line_start, line_end, band_half_thickness)

        frame_proc = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        hp, wp = frame_proc.shape[:2]

        line_start_proc = ratio_to_point(config["line_start_ratio"], wp, hp)
        line_end_proc = ratio_to_point(config["line_end_ratio"], wp, hp)
        band_polygon_proc = build_line_band(
            line_start_proc,
            line_end_proc,
            max(20, int(band_half_thickness * PROCESS_SCALE))
        )

        scale_x = w / wp
        scale_y = h / hp

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

        with estado.lock:
            prev_gray = estado.prev_gray.copy() if estado.prev_gray is not None else None

        best_rect = None
        best_center = None

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [band_polygon_proc], 255)
            thresh = cv2.bitwise_and(thresh, mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_in_gate = False
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA:
                    continue

                x, y, rw, rh = cv2.boundingRect(cnt)

                x = int(x * scale_x)
                y = int(y * scale_y)
                rw = int(rw * scale_x)
                rh = int(rh * scale_y)

                cx = int(x + rw / 2)
                cy = int(y + rh / 2)

                rect = (x, y, rw, rh)
                intersects_band = rect_intersects_band(rect, band_polygon)
                intersects_line = rect_intersects_line(rect, line_start, line_end, tolerance=22)

                if intersects_band and intersects_line and rw >= MIN_WIDTH and rh >= MIN_HEIGHT:
                    detected_in_gate = True
                    if area > best_area:
                        best_area = area
                        best_rect = rect
                        best_center = (cx, cy)

            with estado.lock:
                if detected_in_gate:
                    estado.empty_frames = 0
                    if not estado.gate_occupied:
                        estado.gate_occupied = True
                        estado.count += 1
                        ts = time.time()
                        estado.events.append(ts)
                        estado.last_update = ts
                else:
                    estado.empty_frames += 1
                    if estado.empty_frames >= RESET_FRAMES:
                        estado.gate_occupied = False

        with estado.lock:
            estado.prev_gray = gray.copy()

        snapshot = estado.get_snapshot()

        frame_saida = draw_overlay(
            frame,
            estado.nome_rodovia,
            band_polygon,
            line_start,
            line_end,
            snapshot,
            best_rect,
            best_center
        )

        estado.set_frame(frame_saida)
        time.sleep(0.01)


def gerar_mjpeg(key):
    estado = obter_estado(key)

    while True:
        with estado.lock:
            frame = estado.last_frame.copy() if estado.last_frame is not None else None

        if frame is None:
            vazio = np.zeros((480, 900, 3), dtype=np.uint8)
            cv2.putText(vazio, f"Aguardando camera: {estado.nome_rodovia}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            frame = vazio

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        time.sleep(MJPEG_SLEEP)


@app.on_event("startup")
def startup_event():
    for key, config in CONFIG_RODOVIAS.items():
        t1 = threading.Thread(
            target=capturar_camera,
            args=(key, config),
            daemon=True
        )
        t1.start()

        t2 = threading.Thread(
            target=processar_camera,
            args=(key, config),
            daemon=True
        )
        t2.start()


@app.get("/rodovias")
def listar_rodovias():
    return JSONResponse({
        "rodovias": [
            {
                "key": key,
                "nome": config["nome"]
            }
            for key, config in CONFIG_RODOVIAS.items()
        ]
    })


@app.get("/status/{key}")
def status(key: str):
    estado = obter_estado(key)
    return JSONResponse(estado.get_snapshot())


@app.get("/config/{key}")
def config(key: str, target_x: int | None = None, margin: float | None = None):
    estado = obter_estado(key)
    estado.set_params(target_x=target_x, margin=margin)
    return JSONResponse(estado.get_snapshot())


@app.get("/video/{key}")
def video(key: str):
    obter_estado(key)
    return StreamingResponse(
        gerar_mjpeg(key),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def home():
    opcoes = "".join(
        f'<option value="{key}">{config["nome"]}</option>'
        for key, config in CONFIG_RODOVIAS.items()
    )

    primeira_key = next(iter(CONFIG_RODOVIAS.keys()))

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Monitor Fluxo Rodovias</title>
        <style>
            body {{ font-family: Arial, sans-serif; background:#111; color:#eee; margin:20px; }}
            .wrap {{ display:flex; gap:20px; align-items:flex-start; }}
            .painel {{ min-width:360px; background:#1b1b1b; padding:16px; border-radius:10px; }}
            .linha {{ margin:8px 0; font-size:18px; }}
            .valor {{ font-weight:bold; color:#7CFC00; }}
            img {{ max-width:1000px; border:2px solid #444; border-radius:10px; }}
            input, select {{ width:220px; padding:6px; margin-left:8px; }}
            button {{ padding:8px 12px; cursor:pointer; }}
        </style>
    </head>
    <body>
        <h2>Monitor Fluxo Rodovias</h2>
        <div class="wrap">
            <div>
                <div style="margin-bottom:10px;">
                    <label>
                        Rodovia:
                        <select id="rodoviaSelect" onchange="trocarRodovia()">
                            {opcoes}
                        </select>
                    </label>
                </div>
                <img id="videoFeed" src="/video/{primeira_key}" />
            </div>

            <div class="painel">
                <div class="linha">Rodovia: <span id="nome_rodovia" class="valor">-</span></div>
                <div class="linha">Contagem total: <span id="count_total" class="valor">0</span></div>
                <div class="linha">5 min: <span id="veiculos_5min" class="valor">0</span></div>
                <div class="linha">10 min: <span id="veiculos_10min" class="valor">0</span></div>
                <div class="linha">15 min: <span id="veiculos_15min" class="valor">0</span></div>
                <div class="linha">Rate 5: <span id="rate5" class="valor">0</span></div>
                <div class="linha">Rate 10: <span id="rate10" class="valor">0</span></div>
                <div class="linha">Rate 15: <span id="rate15" class="valor">0</span></div>
                <div class="linha">Forecast 5m: <span id="forecast_5m" class="valor">0</span></div>
                <div class="linha">Sinal: <span id="signal" class="valor">-</span></div>
                <div class="linha">Tendência: <span id="trend" class="valor">-</span></div>
                <div class="linha">Rodando: <span id="running" class="valor">false</span></div>

                <hr>

                <div class="linha">
                    Meta X:
                    <input id="target_x" type="number" value="8" />
                </div>
                <div class="linha">
                    Margem:
                    <input id="margin" type="number" step="0.1" value="3" />
                </div>
                <button onclick="salvarConfig()">Salvar</button>
            </div>
        </div>

        <script>
            function getKeyAtual() {{
                return document.getElementById('rodoviaSelect').value;
            }}

            function trocarRodovia() {{
                const key = getKeyAtual();
                document.getElementById('videoFeed').src = '/video/' + key;
                atualizar();
            }}

            async function atualizar() {{
                const key = getKeyAtual();
                const r = await fetch('/status/' + key);
                const d = await r.json();

                document.getElementById('nome_rodovia').textContent = d.nome_rodovia;
                document.getElementById('count_total').textContent = d.count_total;
                document.getElementById('veiculos_5min').textContent = d.veiculos_5min;
                document.getElementById('veiculos_10min').textContent = d.veiculos_10min;
                document.getElementById('veiculos_15min').textContent = d.veiculos_15min;
                document.getElementById('rate5').textContent = d.rate5;
                document.getElementById('rate10').textContent = d.rate10;
                document.getElementById('rate15').textContent = d.rate15;
                document.getElementById('forecast_5m').textContent = d.forecast_5m;
                document.getElementById('signal').textContent = d.signal;
                document.getElementById('trend').textContent = d.trend;
                document.getElementById('running').textContent = d.running;
                document.getElementById('target_x').value = d.target_x;
                document.getElementById('margin').value = d.margin;
            }}

            async function salvarConfig() {{
                const key = getKeyAtual();
                const target_x = document.getElementById('target_x').value;
                const margin = document.getElementById('margin').value;
                await fetch('/config/' + key + '?target_x=' + target_x + '&margin=' + margin);
                await atualizar();
            }}

            setInterval(atualizar, 2000);
            atualizar();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)