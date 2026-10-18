package ai.executorch.hackathon

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import java.io.File

private const val TAG = "LlmEngine"

/** A token, or the final stats line, streamed out of a generation. */
sealed interface Token {
    data class Text(val text: String) : Token
    data class Stats(val tokensPerSecond: Double, val raw: String) : Token
    data class Failed(val code: Int, val message: String) : Token
}

/**
 * Wraps [LlmModule] in something coroutine-shaped.
 *
 * The native generate() call blocks its thread and pushes tokens through a
 * callback, so it's bridged to a Flow on [Dispatchers.IO]. Calling it from the
 * main thread freezes the UI for the whole generation.
 */
class LlmEngine {

    private var module: LlmModule? = null
    private var loadedPath: String? = null

    val isLoaded: Boolean get() = module != null

    /**
     * Loads a .pte. Blocking — call from a background dispatcher.
     * @return milliseconds spent loading.
     */
    fun load(modelPath: String, tokenizerPath: String): Long {
        require(File(modelPath).exists()) {
            "No model at $modelPath — push it first:  make push MODEL=<preset>"
        }
        require(File(tokenizerPath).exists()) {
            "No tokenizer at $tokenizerPath — the runner needs it beside the model."
        }

        close()

        val config = LlmModuleConfig.create()
            .modulePath(modelPath)
            .tokenizerPath(tokenizerPath)
            .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
            // mmap keeps resident memory down, which matters for a 789MB .pte.
            .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
            // PromptFormat emits BOS itself; letting the module add another
            // shifts every position by one and quietly degrades output.
            .numBos(0)
            .numEos(0)
            .build()

        val started = System.currentTimeMillis()
        module = LlmModule(config).also { it.load() }
        loadedPath = modelPath
        val elapsed = System.currentTimeMillis() - started
        Log.i(TAG, "loaded $modelPath in ${elapsed}ms")
        return elapsed
    }

    /**
     * Streams a response. Stop markers are filtered out of the visible text so
     * the UI doesn't show `<|eot_id|>` at the end of every answer.
     */
    fun generate(
        prompt: String,
        format: PromptFormat,
        maxNewTokens: Int = 256,
        temperature: Float = 0.8f,
    ): Flow<Token> = callbackFlow {
        val llm = module ?: run {
            trySend(Token.Failed(-1, "No model loaded."))
            close()
            return@callbackFlow
        }

        val stops = format.stopTokens()
        val config = LlmGenerationConfig.create()
            .maxNewTokens(maxNewTokens)
            .temperature(temperature)
            .echo(false)
            .build()

        val callback = object : LlmCallback {
            override fun onResult(result: String) {
                if (stops.any { it in result }) {
                    var cleaned = result
                    stops.forEach { cleaned = cleaned.replace(it, "") }
                    if (cleaned.isNotEmpty()) trySend(Token.Text(cleaned))
                    llm.stop()
                    return
                }
                trySend(Token.Text(result))
            }

            override fun onStats(stats: String) {
                trySend(Token.Stats(parseTokensPerSecond(stats), stats))
            }

            override fun onError(code: Int, message: String) {
                Log.e(TAG, "generate failed ($code): $message")
                trySend(Token.Failed(code, message))
            }
        }

        try {
            llm.generate(prompt, config, callback)
        } catch (e: Exception) {
            trySend(Token.Failed(-1, e.message ?: e.toString()))
        }
        close()
        awaitClose { runCatching { llm.stop() } }
    }.flowOn(Dispatchers.IO)

    /** Clears the KV cache. Without this the next prompt inherits this one's context. */
    fun resetContext() {
        runCatching { module?.resetContext() }
    }

    fun stop() {
        runCatching { module?.stop() }
    }

    fun close() {
        runCatching { module?.close() }
        module = null
        loadedPath = null
    }

    /** The runtime reports timings as a JSON blob; pull generated tok/s out of it. */
    private fun parseTokensPerSecond(statsJson: String): Double = runCatching {
        val json = JSONObject(statsJson)
        val generated = json.optLong("generated_tokens", 0L)
        val start = json.optLong("model_execution_start_ms", 0L)
        val inferenceEnd = json.optLong("inference_end_ms", 0L)
        val firstToken = json.optLong("first_token_ms", start)
        val elapsedMs = (inferenceEnd - firstToken).coerceAtLeast(1L)
        if (generated <= 0) 0.0 else generated * 1000.0 / elapsedMs
    }.getOrDefault(0.0)
}
