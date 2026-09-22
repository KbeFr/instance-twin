# **************************************************************************
# * twin_gui.py -- read-only shadow-twin viewer
# *
# * Reads InstanceBase's own public attributes directly at request time --
# * no snapshot method, lock, or buffering added to the twin itself. The
# * twin binds, links, and receives missions through the real instantiate/
# * mission handshakes; this console only displays that state, it never
# * drives it.
# **************************************************************************
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    import core_msgs
    SHARED_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(core_msgs.__file__)), "gui")
except Exception:
    SHARED_STATIC_DIR = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def start_gui(twin, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """Start the viewer in a daemon thread and return the server."""
    handler = _make_handler(twin)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="twin-gui", daemon=True).start()
    logger.info("twin viewer listening on http://%s:%d", host, port)
    return server


def _lifecycle(twin) -> str:
    if twin.agent is None:
        return "UNBOUND"
    return "LIVE" if twin.agent.linked else "BOUND"


def _state_dict(twin) -> dict:
    agent = twin.agent
    mission = twin.active_mission

    data: dict[str, Any] = {
        "twin_name": twin.name,
        "namespace": twin.namespace,
        "lifecycle": _lifecycle(twin),
        "sim_time": round(twin.sim_time, 3),
        "stop_flag": twin.stop_flag,
        "arrive_flag": twin.arrive_flag,
        "goal_threshold": twin.goal_threshold,
        "obstacles": [
            {
                "id": o.id,
                "x": float(o.x), "y": float(o.y),
                "radius": float(o.radius) if o.radius is not None else 0.0,
                "vx": float(o.vx), "vy": float(o.vy),
            }
            for o in twin.known_obstacles.values()
        ],
        "trajectory": [[float(p[0, 0]), float(p[1, 0])] for p in twin.trajectory[-500:]],
        "agent": None,
        "mission": None,
        "goal": None,
    }

    if agent is not None:
        state = agent.state.reshape(-1).tolist()
        data["agent"] = {
            "name": agent.name,
            "kind": getattr(agent.kind, "value", str(agent.kind)),
            "linked": agent.linked,
            "radius": agent.radius,
            "kinematics": agent.kinematics.__class__.__name__ if agent.kinematics else None,
            "controller": agent.controller.__class__.__name__ if agent.controller else None,
            "x": float(state[0]) if state else 0.0,
            "y": float(state[1]) if len(state) > 1 else 0.0,
            "theta": float(state[2]) if len(state) > 2 else 0.0,
            "velocity": [round(float(v), 4) for v in agent.velocity.reshape(-1).tolist()],
            "battery_pct": round(float(agent.battery.status), 2) if agent.battery else 100.0,
            "battery_depleted": agent.battery_depleted,
        }
        if agent.goal is not None:
            data["goal"] = [float(agent.goal[0, 0]), float(agent.goal[1, 0])]

    if mission is not None:
        data["mission"] = {
            "mission_id": mission.mission_id,
            "type": mission.mission_type.name,
            "posture": mission.mission_posture.name,
            "status": mission.mission_status.name,
            "goal_xy": list(mission.goal_xy) if mission.goal_xy else None,
            "waypoints": [list(w) for w in mission.waypoints],
        }

    return data


def _make_handler(twin):

    class TwinViewerHandler(BaseHTTPRequestHandler):
        server_version = "TwinViewer/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.debug("gui %s", fmt % args)

        def _respond(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str) -> None:
            """Serve a shared asset (currently just theme.css) from core_msgs."""
            path = os.path.normpath(os.path.join(SHARED_STATIC_DIR, name))
            if not path.startswith(SHARED_STATIC_DIR) or not os.path.isfile(path):
                return self._respond(404, b'{"error":"not found"}', "application/json; charset=utf-8")
            ctype = {".css": "text/css", ".js": "text/javascript"}.get(
                os.path.splitext(path)[1], "text/plain")
            with open(path, "rb") as fh:
                self._respond(200, fh.read(), f"{ctype}; charset=utf-8")

        def do_GET(self):                                   # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/":
                self._respond(200, CONSOLE_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/healthz":
                body = json.dumps({"status": "ok", "lifecycle": _lifecycle(twin),
                                   "twin": twin.twin_name}).encode()
                self._respond(200, body, "application/json; charset=utf-8")
            elif path == "/api/state":
                self._respond(200, json.dumps(_state_dict(twin), default=str).encode(),
                              "application/json; charset=utf-8")
            elif path.endswith((".css", ".js")):
                self._static(os.path.basename(path))
            else:
                self._respond(404, b'{"error":"not found"}', "application/json; charset=utf-8")

    return TwinViewerHandler


CONSOLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shadow twin viewer</title>
<link rel="stylesheet" href="/theme.css">
</head>
<body>
<header>
  <h1 id="twinName">—<span id="ns"></span></h1>
  <span class="meter">sim <i id="simTime">0.0</i>s</span>
  <div class="rail" id="rail">
    <b data-s="UNBOUND">unbound</b><b data-s="BOUND">bound</b><b data-s="LIVE">live</b>
  </div>
</header>
<div class="banner" id="banner"></div>

<main class="active" style="grid-template-columns:minmax(0,1fr) 300px">
  <div class="field">
    <canvas id="cv"></canvas>
    <div class="legend">
      <span><s style="background:var(--signal)"></s>twin</span>
      <span><s style="background:var(--goal)"></s>goal</span>
      <span><s style="background:var(--warn)"></s>obstacle</span>
    </div>
    <div class="readout" id="readout"></div>
  </div>

  <aside>
    <section>
      <span class="eyebrow">Agent</span>
      <div id="agentBox"><p class="empty">No agent bound. Waiting for discovery + instantiate.</p></div>
    </section>
    <section>
      <span class="eyebrow">Mission</span>
      <div id="missionBox"><p class="empty">No active mission.</p></div>
    </section>
    <section>
      <span class="eyebrow">Obstacles</span>
      <div id="obsBox"><p class="empty">None observed.</p></div>
    </section>
  </aside>
</main>

<script>
const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
let snap = null, lastError = 0;

function banner(msg){
  const b = $('banner');
  b.className = 'banner' + (msg ? ' show' : '');
  if(msg) b.textContent = msg;
}

function fit(s, w, h){
  const pts = [...s.trajectory];
  if(s.agent) pts.push([s.agent.x, s.agent.y]);
  if(s.goal) pts.push(s.goal);
  s.obstacles.forEach(o => pts.push([o.x+o.radius, o.y+o.radius], [o.x-o.radius, o.y-o.radius]));
  if(!pts.length) pts.push([0,0]);
  let xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  let cx = (Math.min(...xs)+Math.max(...xs))/2, cy = (Math.min(...ys)+Math.max(...ys))/2;
  let span = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys), 6) * 1.25;
  const scale = Math.min(w, h) / span;
  return {toX: x => w/2 + (x-cx)*scale, toY: y => h/2 - (y-cy)*scale, scale, cx, cy, span};
}

function draw(){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if(!snap) return;
  const t = fit(snap, w, h);

  const step = t.span > 40 ? 10 : t.span > 16 ? 5 : 1;
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(127,209,222,.07)';
  ctx.beginPath();
  for(let g = Math.floor((t.cx - t.span)/step)*step; g < t.cx + t.span; g += step){
    const x = t.toX(g); ctx.moveTo(x, 0); ctx.lineTo(x, h);
  }
  for(let g = Math.floor((t.cy - t.span)/step)*step; g < t.cy + t.span; g += step){
    const y = t.toY(g); ctx.moveTo(0, y); ctx.lineTo(w, y);
  }
  ctx.stroke();
  ctx.strokeStyle = 'rgba(127,209,222,.18)';
  ctx.beginPath();
  ctx.moveTo(t.toX(0),0); ctx.lineTo(t.toX(0),h);
  ctx.moveTo(0,t.toY(0)); ctx.lineTo(w,t.toY(0)); ctx.stroke();

  snap.obstacles.forEach(o => {
    ctx.beginPath();
    ctx.arc(t.toX(o.x), t.toY(o.y), Math.max(o.radius*t.scale, 3), 0, Math.PI*2);
    ctx.fillStyle = 'rgba(224,160,64,.16)'; ctx.fill();
    ctx.strokeStyle = 'rgba(224,160,64,.75)'; ctx.lineWidth = 1.2; ctx.stroke();
  });

  if(snap.goal){
    const gx = t.toX(snap.goal[0]), gy = t.toY(snap.goal[1]);
    ctx.strokeStyle = 'rgba(217,123,166,.85)'; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(gx, gy, snap.goal_threshold*t.scale, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(gx-7,gy); ctx.lineTo(gx+7,gy); ctx.moveTo(gx,gy-7); ctx.lineTo(gx,gy+7);
    ctx.stroke();
  }

  const tr = snap.trajectory;
  if(tr.length > 1){
    ctx.lineWidth = 1.6; ctx.lineCap = 'round';
    for(let i=1;i<tr.length;i++){
      ctx.strokeStyle = `rgba(127,209,222,${0.06 + 0.5*(i/tr.length)})`;
      ctx.beginPath();
      ctx.moveTo(t.toX(tr[i-1][0]), t.toY(tr[i-1][1]));
      ctx.lineTo(t.toX(tr[i][0]), t.toY(tr[i][1]));
      ctx.stroke();
    }
  }

  if(snap.agent){
    const a = snap.agent, x = t.toX(a.x), y = t.toY(a.y);
    const r = Math.max(a.radius*t.scale, 5);
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle = snap.stop_flag ? 'rgba(224,104,95,.25)' : 'rgba(127,209,222,.22)';
    ctx.fill();
    ctx.strokeStyle = snap.stop_flag ? '#e0685f' : '#7fd1de';
    ctx.lineWidth = 1.6; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x,y);
    ctx.lineTo(x + Math.cos(a.theta)*r*1.9, y - Math.sin(a.theta)*r*1.9);
    ctx.stroke();
    if(!a.linked){
      ctx.setLineDash([3,4]); ctx.strokeStyle = 'rgba(224,160,64,.8)';
      ctx.beginPath(); ctx.arc(x,y,r+6,0,Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
    }
    $('readout').innerHTML =
      `x <em>${a.x.toFixed(2)}</em>  y <em>${a.y.toFixed(2)}</em>  θ <em>${a.theta.toFixed(2)}</em><br>` +
      `v <em>${(a.velocity[0]??0).toFixed(2)}</em>  ω <em>${(a.velocity[1]??0).toFixed(2)}</em>` +
      `  ·  1 grid = ${step} m`;
  } else {
    $('readout').textContent = 'No shadow model — the field is empty until an agent binds.';
  }
}

function kv(pairs){
  return '<dl>' + pairs.map(([k,val]) => `<dt>${k}</dt><dd>${val}</dd>`).join('') + '</dl>';
}

function render(s){
  snap = s;
  $('twinName').innerHTML = `${s.twin_name} <span>· ${s.namespace}</span>`;
  $('simTime').textContent = s.sim_time.toFixed(1);
  document.querySelectorAll('#rail b').forEach(el => {
    el.className = el.dataset.s === s.lifecycle
      ? 'on' + (s.lifecycle === 'BOUND' ? ' bound' : '') : '';
  });

  const a = s.agent;
  $('agentBox').innerHTML = a ? kv([
    ['name', a.name], ['kind', a.kind], ['kinematics', a.kinematics],
    ['controller', a.controller || '—'], ['radius', a.radius.toFixed(2) + ' m'],
    ['agent link', a.linked ? '<span class="ok">live</span>' : '<span class="warn">awaiting pose</span>'],
    ['battery', a.battery_depleted ? '<span class="error">depleted</span>' : a.battery_pct.toFixed(1) + ' %'],
  ]) : '<p class="empty">No agent bound. Waiting for discovery + instantiate.</p>';

  const m = s.mission;
  $('missionBox').innerHTML = m ? kv([
    ['id', m.mission_id], ['type', m.type], ['posture', m.posture],
    ['status', `<span class="${m.status==='COMPLETE'?'ok':'info'}">${m.status}</span>`],
    ['goal', m.goal_xy ? m.goal_xy.map(n=>n.toFixed(2)).join(', ') : '—'],
    ['waypoints', m.waypoints.length || '—'],
    ['arrived', s.arrive_flag ? '<span class="ok">yes</span>' : 'no'],
  ]) : '<p class="empty">No active mission.</p>';

  $('obsBox').innerHTML = s.obstacles.length
    ? '<dl>' + s.obstacles.map(o =>
        `<dt>${o.id}</dt><dd>${o.x.toFixed(2)}, ${o.y.toFixed(2)} (r=${o.radius.toFixed(2)})</dd>`).join('') + '</dl>'
    : '<p class="empty">None observed.</p>';

  draw();
}

async function poll(){
  try{
    const r = await fetch('/api/state');
    render(await r.json());
    if(lastError){ banner(''); lastError = 0; }
  }catch(e){
    lastError = 1;
    banner('Lost the twin — the container may have stopped.');
  }
}

addEventListener('resize', draw);
poll();
setInterval(poll, 200);
</script>
</body>
</html>
"""