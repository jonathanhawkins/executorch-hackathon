#!/usr/bin/env bash
# Start (or stop) the on-device camera + vision demo on the VENTUNO Q.
#
#   ./scripts/board_web.sh start     # dev server, fastest to get running
#   ./scripts/board_web.sh build     # production build, then serve
#   ./scripts/board_web.sh stop
#   ./scripts/board_web.sh logs
#   ./scripts/board_web.sh push      # re-push app sources after editing
#
# Everything runs on the board: camera, NPU inference and the web server.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. ./scripts/android_env.sh

APP=/home/arduino/camvlm
NODE=/home/arduino/node/bin
CAM_DIR=/dev/shm/cam
CMD="${1:-start}"

adb get-state >/dev/null 2>&1 || { echo "No device on adb." >&2; exit 1; }

board_ip() {
  adb shell "hostname -I 2>/dev/null | awk '{print \$1}'" | tr -d '\r\n'
}

start_camera() {
  # pkill -f would match this very command line (it contains "gst-launch"),
  # so match the executable name exactly instead.
  adb shell "
    mkdir -p $CAM_DIR
    pgrep -x gst-launch-1.0 >/dev/null && exit 0
    setsid gst-launch-1.0 -q v4l2src device=/dev/video0 ! videoconvert ! \
      jpegenc quality=70 ! multifilesink location=$CAM_DIR/f%05d.jpg max-files=4 \
      > $CAM_DIR/gst.log 2>&1 < /dev/null &
  " >/dev/null 2>&1 || true
  sleep 3
}

case "$CMD" in
  push)
    echo "==> Pushing app sources"
    adb shell "mkdir -p $APP/app/api/snapshot $APP/app/api/analyze $APP/app/api/status $APP/app/api/transcript $APP/app/api/billboard $APP/lib"
    adb push web/package.json web/next.config.mjs "$APP/" >/dev/null
    adb push web/lib/board.js "$APP/lib/" >/dev/null
    adb push web/app/layout.jsx web/app/page.jsx web/app/globals.css "$APP/app/" >/dev/null
    adb push web/app/api/snapshot/route.js "$APP/app/api/snapshot/" >/dev/null
    adb push web/app/api/analyze/route.js "$APP/app/api/analyze/" >/dev/null
    adb push web/app/api/status/route.js "$APP/app/api/status/" >/dev/null
    adb push web/app/api/transcript/route.js "$APP/app/api/transcript/" >/dev/null
    adb push web/app/api/billboard/route.js "$APP/app/api/billboard/" >/dev/null
    adb shell "mkdir -p /home/arduino/bb/scripts /home/arduino/bb/vendor/face_parsing /home/arduino/bb/vendor/ghost /home/arduino/bb/assets"
    adb push scripts/board_billboard_daemon.py scripts/face_detect.py scripts/face_matte.py scripts/make_face_raw.py scripts/face_swap.py /home/arduino/bb/scripts/ >/dev/null
    # Placement, and the body-clip path the tryon-clip mode uses.
    adb push scripts/garment_overlay.py scripts/garment_clip.py scripts/person_seg.py \
      scripts/make_person_raw.py scripts/person_matte.py /home/arduino/bb/scripts/ >/dev/null
    # The GHOST swap backend (NPU, Apache-2.0) and what it imports.
    adb push scripts/ghost_booth.py scripts/ghost_swap.py scripts/make_ghost_raw.py /home/arduino/bb/scripts/ >/dev/null
    # warm_runner.py keeps one qnn_executor_runner alive per model; without
    # it the daemon falls back to relaunching per inference.
    adb push scripts/warm_runner.py /home/arduino/bb/scripts/ >/dev/null
    # ControlNet try-on: the generate mode's module, the diffusion driver it
    # shells out to, and the pre-tokenized garment catalogue.
    adb push scripts/board_tryon_gen.py scripts/board_sd_driver.py /home/arduino/bb/scripts/ >/dev/null
    if [ -d .tryon_prompts ]; then
      adb shell "mkdir -p /home/arduino/bb/prompts" >/dev/null
      adb push .tryon_prompts/. /home/arduino/bb/prompts/ >/dev/null
    fi
    adb push vendor/ghost/. /home/arduino/bb/vendor/ghost/ >/dev/null
    # Face swap needs the product photo with the model's head still on it,
    # not the headless *.template.png the compositing path uses.
    adb shell "mkdir -p /home/arduino/bb/templates" >/dev/null
    adb push assets/templates/. /home/arduino/bb/templates/ >/dev/null
    adb push vendor/face_parsing/. /home/arduino/bb/vendor/face_parsing/ >/dev/null
    adb push web/public/billboard/. /home/arduino/bb/assets/ >/dev/null
    adb push scripts/make_vision_raw.py /home/arduino/make_vision_raw.py >/dev/null
    adb push scripts/board_mic_daemon.py /home/arduino/mic_daemon.py >/dev/null
    echo "    done — restart with: $0 start"
    ;;

  stop)
    echo "==> Stopping web server and camera"
    adb shell "pkill -f 'next-server' 2>/dev/null; pkill -f 'next dev' 2>/dev/null; pkill -x gst-launch-1.0 2>/dev/null; true" >/dev/null 2>&1 || true
    echo "    stopped"
    ;;

  logs)
    adb shell "tail -40 $APP/server.log 2>/dev/null" | tr -d '\r'
    ;;

  build)
    start_camera
    echo "==> Building (slow on this CPU — several minutes)"
    adb shell "export PATH=$NODE:\$PATH; cd $APP && npm run build 2>&1 | tail -20" | tr -d '\r'
    echo "==> Serving"
    # Same detaching trick as `start`: an inline `setsid ... &` inside
    # `adb shell "..."` dies when that shell exits, so the production server
    # has to be launched from a script file on the board too.
    adb shell "cat > $APP/serve.sh <<'SERVEEOF'
#!/usr/bin/env bash
export PATH=$NODE:\$PATH
cd $APP
exec npm run start
SERVEEOF
chmod +x $APP/serve.sh"
    adb shell "pkill -f next-server 2>/dev/null; true" >/dev/null 2>&1 || true
    # `disown`, and a separate adb shell from the one that wrote the script —
    # both are load-bearing, same as in `start`.
    adb shell "setsid $APP/serve.sh > $APP/server.log 2>&1 < /dev/null & disown; sleep 1" \
      >/dev/null 2>&1 || true
    sleep 6
    echo "    http://$(board_ip):3000"
    ;;

  start)
    start_camera
    echo "==> Starting dev server on the board"
    # Detaching has to go through a script file on the board. An inline
    # `setsid ... &` inside `adb shell "..."` dies when the shell exits.
    adb shell "cat > $APP/run.sh <<'RUNEOF'
#!/usr/bin/env bash
export PATH=$NODE:\$PATH
cd $APP
exec npm run dev
RUNEOF
chmod +x $APP/run.sh"
    adb shell "pkill -f 'next dev' 2>/dev/null; pkill -f next-server 2>/dev/null; true" >/dev/null 2>&1 || true
    adb shell "setsid $APP/run.sh > $APP/server.log 2>&1 < /dev/null & disown; sleep 1" >/dev/null 2>&1 || true
    echo "    waiting for it to come up..."
    for _ in $(seq 1 30); do
      if adb shell "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/api/status 2>/dev/null" | tr -d '\r' | grep -q 200; then
        break
      fi
      sleep 2
    done
    IP="$(board_ip)"
    echo
    echo "    Open:  http://${IP}:3000"
    echo "    Logs:  $0 logs"
    adb shell "curl -s http://127.0.0.1:3000/api/status" | tr -d '\r'
    echo
    ;;

  *)
    echo "usage: $0 {start|build|stop|logs|push}" >&2
    exit 2
    ;;
esac
