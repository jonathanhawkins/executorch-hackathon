# ExecuTorch hackathon — common tasks.
# Everything runs inside .venv; no need to activate it yourself.

VENV    := .venv
PY      := $(VENV)/bin/python
PATH_   := $(CURDIR)/$(VENV)/bin:$(PATH)
MODEL   ?= smollm2-135m
PROMPT  ?= Explain what on-device AI is, in two sentences.
# Where the AI Hub ControlNet-Canny context binaries live on the board.
BOARD_CN_CTX ?= /home/arduino/controlnet/extracted/controlnet_canny-qnn_context_binary-w8a16-qualcomm_qcs8275

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Override the model:  make export MODEL=qwen2.5-0.5b"

.PHONY: setup
setup: ## Create the venv and install everything
	./scripts/setup.sh

.PHONY: list
list: ## List the model presets
	@$(PY) scripts/export_model.py --list

.PHONY: export
export: ## Export MODEL to a .pte
	@PATH="$(PATH_)" $(PY) scripts/export_model.py $(MODEL)

.PHONY: run
run: ## Run MODEL's .pte and report tok/s
	@PATH="$(PATH_)" $(PY) scripts/run_model.py $(MODEL) --prompt "$(PROMPT)"

.PHONY: smoke
smoke: ## Export + run the tiny model, end to end
	@$(MAKE) export MODEL=smollm2-135m
	@$(MAKE) run MODEL=smollm2-135m

.PHONY: push
push: ## adb push MODEL's .pte to the Android device
	@./scripts/push_android.sh $(MODEL)

.PHONY: apk
apk: ## Build the Android demo APK
	@cd android && . ../scripts/android_env.sh && ./gradlew :app:assembleDebug
	@echo "  android/app/build/outputs/apk/debug/app-debug.apk"

.PHONY: install
install: apk ## Build, install, and launch the demo app on the device
	@cd android && . ../scripts/android_env.sh && ./gradlew :app:installDebug
	@. ./scripts/android_env.sh && adb shell monkey -p ai.executorch.hackathon \
		-c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
	@echo "  launched ai.executorch.hackathon"

.PHONY: deploy
deploy: push install ## push MODEL + build/install the app, in one go

.PHONY: logcat
logcat: ## Tail the app's logs (model load errors show up here)
	@. ./scripts/android_env.sh && adb logcat -s LlmEngine ExecuTorch LlmModule AndroidRuntime

# --- Arduino VENTUNO Q (IoT track) -----------------------------------------
# The board is aarch64 Ubuntu over adb, NOT Android — it runs the .pte through
# Python, the same way the Mac does. No APK involved.

.PHONY: board-setup
board-setup: ## Install the ExecuTorch runtime on the board (one-off, slow)
	@. ./scripts/android_env.sh && adb push scripts/board_setup.sh /home/arduino/board_setup.sh
	@. ./scripts/android_env.sh && adb shell "chmod +x ~/board_setup.sh && bash ~/board_setup.sh"

.PHONY: board-run
board-run: ## Push MODEL to the board and run it there
	@./scripts/board_run.sh $(MODEL) "$(PROMPT)"

.PHONY: board-info
board-info: ## Show what the attached board actually is
	@. ./scripts/android_env.sh && adb shell "uname -a; echo; \
		cat /sys/devices/soc0/machine 2>/dev/null; \
		lscpu | grep -iE 'architecture|^cpu\(s\)|model name|max mhz'; \
		free -h | head -2; df -h / | tail -1" | tr -d '\r'

.PHONY: board-npu
board-npu: ## Check whether the board's Hexagon NPU is reachable (no sudo)
	@./scripts/board_npu_check.sh

.PHONY: board-npu-setup
board-npu-setup: ## Extract the NPU source bundle on the board, print the sudo steps
	@./scripts/board_npu_setup.sh

.PHONY: board-npu-run
board-npu-run: ## Run MODEL on the Hexagon NPU (MODEL=gemma4_e2b)
	@./scripts/board_npu_run.sh $(if $(NPU_MODEL),$(NPU_MODEL),gemma4_e2b) "$(PROMPT)"

.PHONY: board-npu-vision
board-npu-vision: ## Run a VLM on the NPU (optional IMAGE=path, NPU_MODEL=internvl3-1b)
	@./scripts/board_npu_vision.sh $(if $(NPU_MODEL),$(NPU_MODEL),internvl3-1b) "$(IMAGE)" "$(PROMPT)"

.PHONY: board-whisper-export
board-whisper-export: ## Export Whisper to a QNN .pte on the board (slow, ~20 min)
	@./scripts/board_whisper_export.sh

.PHONY: board-whisper
board-whisper: ## Transcribe AUDIO on the Hexagon NPU (AUDIO=clip.wav)
	@./scripts/board_whisper_run.sh "$(AUDIO)"

.PHONY: web
web: ## Start the camera + vision web demo on the board
	@./scripts/board_web.sh start

.PHONY: web-push
web-push: ## Re-push the web app sources to the board
	@./scripts/board_web.sh push

.PHONY: web-stop
web-stop: ## Stop the web demo and camera
	@./scripts/board_web.sh stop

.PHONY: web-logs
web-logs: ## Tail the web server log on the board
	@./scripts/board_web.sh logs

.PHONY: billboard-assets
billboard-assets: ## Regenerate the billboard backdrop/garment/collar
	@$(PY) scripts/make_billboard_assets.py -o web/public/billboard
	@. ./scripts/android_env.sh && adb push web/public/billboard/. /home/arduino/bb/assets/ >/dev/null && echo "    pushed to board"

.PHONY: bb-logs
bb-logs: ## Tail the billboard compositor log on the board
	@. ./scripts/android_env.sh && adb shell "tail -30 /dev/shm/billboard/daemon.log 2>/dev/null" | tr -d '\r'

.PHONY: mic-logs
mic-logs: ## Tail the live-speech listener log on the board
	@. ./scripts/android_env.sh && adb shell "tail -40 /dev/shm/mic/daemon.log 2>/dev/null" | tr -d '\r'

.PHONY: board-faceparse-export
board-faceparse-export: ## Export BiSeNet face parsing to a QNN .pte (~10 min)
	@./scripts/board_faceparse_export.sh

.PHONY: board-faceparse
board-faceparse: ## Cut out a head on the NPU (IMAGE=portrait.jpg [BBOX=x,y,w,h])
	@BBOX="$(BBOX)" ./scripts/board_faceparse_run.sh "$(IMAGE)"

.PHONY: board-personseg-export
board-personseg-export: ## Export person segmentation to a QNN .pte (~10 min)
	@SEG_MODEL="$(SEG_MODEL)" CALIB="$(CALIB)" ./scripts/board_personseg_export.sh

.PHONY: board-personseg
board-personseg: ## Segment the person on the NPU (IMAGE=frame.jpg [CLASSES=1])
	@OUT="$(OUT)" CLASSES="$(CLASSES)" SEG_MODEL="$(SEG_MODEL)" \
		ITERATIONS="$(ITERATIONS)" ./scripts/board_personseg_run.sh "$(IMAGE)"

.PHONY: board-sd-export
board-sd-export: ## Wrap the AI Hub Stable Diffusion context binaries into .pte (~1 min)
	@FORCE="$(FORCE)" CTX="$(CTX)" ARTIFACT="$(ARTIFACT)" ./scripts/board_sd_export.sh

.PHONY: board-sd
board-sd: ## Generate an image on the NPU (PROMPT="..." [STEPS=20] [SEED=42])
	@OUT="$(OUT)" STEPS="$(STEPS)" SEED="$(SEED)" GUIDANCE="$(GUIDANCE)" \
		NEGATIVE="$(NEGATIVE)" SAMPLER="$(SAMPLER)" SCHEDULE="$(SCHEDULE)" \
		PLAIN="$(PLAIN)" PERSISTENT="$(PERSISTENT)" ./scripts/board_sd_run.sh "$(PROMPT)"

.PHONY: board-cn-export
board-cn-export: ## Wrap the ControlNet-Canny context binaries into .pte (~20 s)
	@CTX="$(BOARD_CN_CTX)" ARTIFACT=/home/arduino/models/controlnet-npu \
		STAGES=text_encoder,controlnet,unet,vae PREFIX=cn15 FORCE="$(FORCE)" \
		./scripts/board_sd_export.sh

.PHONY: board-cn
board-cn: ## Generate conditioned on a photo (IMAGE=frame.jpg PROMPT="...")
	@MODEL_DIR=/home/arduino/models/controlnet-npu PREFIX=cn15 BASE=sd15 \
		PREDICTION=epsilon IMAGE="$(IMAGE)" OUT="$(OUT)" STEPS="$(STEPS)" \
		SEED="$(SEED)" GUIDANCE="$(GUIDANCE)" COND_SCALE="$(COND_SCALE)" \
		CANNY_LOW="$(CANNY_LOW)" CANNY_HIGH="$(CANNY_HIGH)" PERSISTENT="$(PERSISTENT)" \
		STATS="$(STATS)" NEGATIVE="$(NEGATIVE)" ./scripts/board_sd_run.sh "$(PROMPT)"

.PHONY: board-cn-tryon
board-cn-tryon: ## Put a person in a described garment (IMAGE=frame.jpg GARMENT="...")
	@OUT="$(OUT)" STEPS="$(STEPS)" SEED="$(SEED)" MATCH_BODY="$(MATCH_BODY)" \
		./scripts/board_cn_tryon.sh "$(IMAGE)" "$(GARMENT)"

.PHONY: board-tryon
board-tryon: ## Overlay a garment (IMAGE=frame.jpg [OCCLUDE=1] [CLIP=1])
	@OUT="$(OUT)" GARMENT="$(GARMENT)" OCCLUDE="$(OCCLUDE)" CLIP="$(CLIP)" \
		./scripts/board_tryon.sh "$(IMAGE)"

.PHONY: faceswap
faceswap: ## Put a face into the product photo (SOURCE=me.jpg [TARGET=...] [OUT=...])
	@PATH="$(PATH_)" $(PY) scripts/face_swap.py --source "$(SOURCE)" \
		$(if $(TARGET),--target "$(TARGET)") $(if $(OUT),--out "$(OUT)") --bench

.PHONY: board-faceswap-export
board-faceswap-export: ## Export GHOST face swap to QNN .pte files (~20 min)
	@QUANT="$(QUANT)" CALIB="$(CALIB)" STAGE="$(STAGE)" ARTIFACT="$(ARTIFACT)" \
		PER_CHANNEL="$(PER_CHANNEL)" ACT_OBSERVER="$(ACT_OBSERVER)" \
		AAD_QUANT="$(AAD_QUANT)" ./scripts/board_faceswap_export.sh

.PHONY: board-faceswap
board-faceswap: ## Swap a face onto the product photo on the NPU (IMAGE=frame.jpg)
	@OUT="$(OUT)" QUANT="$(QUANT)" SCALE="$(SCALE)" ITERATIONS="$(ITERATIONS)" \
		IDENTITY="$(IDENTITY)" REFERENCE="$(REFERENCE)" BACKEND="$(BACKEND)" \
		./scripts/board_faceswap_run.sh "$(IMAGE)" $(TARGET)

.PHONY: ghost-xnnpack-export
ghost-xnnpack-export: ## Export GHOST to an XNNPACK .pte (CPU fallback, on the Mac)
	@PATH="$(PATH_)" $(PY) scripts/ghost_xnnpack_export.py \
		$(if $(STAGE),--stage "$(STAGE)") -o models/ghost

.PHONY: ghost-swap
ghost-swap: ## GHOST face swap in fp32 on the Mac (SOURCE=me.jpg [TARGET=...])
	@PATH="$(PATH_)" $(PY) scripts/ghost_swap.py --source "$(SOURCE)" \
		$(if $(TARGET),--target "$(TARGET)") $(if $(OUT),-o "$(OUT)") --bench

.PHONY: board-toolchain
board-toolchain: ## Install gcc/cmake/ninja on the board, no root needed
	@./scripts/board_toolchain.sh

.PHONY: board-build-runner
board-build-runner: ## Build qnn_executor_runner with the persistent-stdin patch
	@./scripts/board_build_runner.sh

.PHONY: board-shell
board-shell: ## Interactive shell on the board
	@. ./scripts/android_env.sh && adb shell

.PHONY: doctor
doctor: ## Check the toolchain is sane
	@PATH="$(PATH_)" $(PY) scripts/doctor.py

.PHONY: clean
clean: ## Delete generated .pte/.ptd artifacts under models/
	@find models -type f \( -name '*.pte' -o -name '*.ptd' \) -print -delete
