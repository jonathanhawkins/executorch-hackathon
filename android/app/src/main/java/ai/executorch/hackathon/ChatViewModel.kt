package ai.executorch.hackathon

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Where `scripts/push_android.sh` puts things. */
private const val DEVICE_DIR = "/data/local/tmp/llama"

class ChatViewModel : ViewModel() {

    private val engine = LlmEngine()
    private var generateJob: Job? = null

    var modelPath by mutableStateOf("$DEVICE_DIR/model.pte")
    var tokenizerPath by mutableStateOf("$DEVICE_DIR/tokenizer.json")
    var prompt by mutableStateOf("Explain what on-device AI is, in two sentences.")
    var systemPrompt by mutableStateOf("You are a helpful assistant.")
    var format by mutableStateOf(PromptFormat.LLAMA3)
    var output by mutableStateOf("")
        private set
    var status by mutableStateOf("")
        private set
    var isLoaded by mutableStateOf(false)
        private set
    var isGenerating by mutableStateOf(false)
        private set
    var isLoading by mutableStateOf(false)
        private set

    val isBusy: Boolean get() = isLoading || isGenerating

    fun onModelPathChange(path: String) {
        modelPath = path
        // Re-guess the template so switching models doesn't silently keep the
        // wrong framing — the failure mode there looks like a bad model.
        format = PromptFormat.guessFrom(path)
    }

    fun loadModel() {
        if (isBusy) return
        isLoading = true
        status = "Loading ${modelPath.substringAfterLast('/')}…"
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) { engine.load(modelPath, tokenizerPath) }
            }.onSuccess { ms ->
                isLoaded = true
                status = "Loaded in ${ms}ms · template ${format.label}"
            }.onFailure { e ->
                isLoaded = false
                status = "Load failed: ${e.message}"
            }
            isLoading = false
        }
    }

    fun generate() {
        if (!isLoaded || isBusy) return
        isGenerating = true
        engine.resetContext()

        val framed = format.format(systemPrompt, prompt)
        output = ""
        status = "Generating…"

        generateJob = viewModelScope.launch {
            engine.generate(framed, format).collect { token ->
                when (token) {
                    is Token.Text -> output += token.text
                    is Token.Stats ->
                        status = "%.1f tok/s on device".format(token.tokensPerSecond)
                    is Token.Failed ->
                        status = "Error ${token.code}: ${token.message}"
                }
            }
            isGenerating = false
        }
    }

    fun stop() {
        engine.stop()
        generateJob?.cancel()
        isGenerating = false
        status = "Stopped."
    }

    fun clear() {
        output = ""
        status = ""
        engine.resetContext()
    }

    override fun onCleared() {
        engine.close()
        super.onCleared()
    }
}
