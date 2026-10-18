package ai.executorch.hackathon

/**
 * Chat templates, applied on-device.
 *
 * The Python side gets this free from `tokenizer.apply_chat_template`, but
 * [org.pytorch.executorch.extension.llm.LlmModule.generate] takes a raw string,
 * so the app has to build the framing itself. Skip it and an instruct model
 * free-associates instead of answering — the model is fine, the framing is wrong.
 *
 * Each template already carries its own BOS, so the module is configured with
 * `numBos(0)` to avoid emitting it twice.
 */
enum class PromptFormat(val label: String) {

    /** Llama 3.x — `<|begin_of_text|>` + header blocks. */
    LLAMA3("Llama 3.x") {
        override fun format(systemPrompt: String, userPrompt: String): String = buildString {
            append("<|begin_of_text|>")
            if (systemPrompt.isNotBlank()) {
                append("<|start_header_id|>system<|end_header_id|>\n\n")
                append(systemPrompt)
                append("<|eot_id|>")
            }
            append("<|start_header_id|>user<|end_header_id|>\n\n")
            append(userPrompt)
            append("<|eot_id|>")
            append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }

        override fun stopTokens() = listOf("<|eot_id|>", "<|end_of_text|>")
    },

    /** Qwen 2.x and most ChatML-style models. */
    CHATML("Qwen / ChatML") {
        override fun format(systemPrompt: String, userPrompt: String): String = buildString {
            if (systemPrompt.isNotBlank()) {
                append("<|im_start|>system\n").append(systemPrompt).append("<|im_end|>\n")
            }
            append("<|im_start|>user\n").append(userPrompt).append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }

        override fun stopTokens() = listOf("<|im_end|>", "<|endoftext|>")
    },

    /** Gemma 3 — no system role; the system prompt is folded into the user turn. */
    GEMMA("Gemma 3") {
        override fun format(systemPrompt: String, userPrompt: String): String = buildString {
            append("<bos><start_of_turn>user\n")
            if (systemPrompt.isNotBlank()) append(systemPrompt).append("\n\n")
            append(userPrompt)
            append("<end_of_turn>\n<start_of_turn>model\n")
        }

        override fun stopTokens() = listOf("<end_of_turn>", "<eos>")
    },

    /**
     * Gemma 4 — different turn markers from Gemma 3, which is easy to miss.
     * Verified against the chat_template.jinja shipped with the QCS8275 build
     * ("Google Gemma 4 Canonical Chat Template") and the runner's own output.
     */
    GEMMA4("Gemma 4") {
        override fun format(systemPrompt: String, userPrompt: String): String = buildString {
            append("<bos><|turn>user\n")
            if (systemPrompt.isNotBlank()) append(systemPrompt).append("\n\n")
            append(userPrompt)
            append("<turn|>\n<|turn>model\n")
        }

        override fun stopTokens() = listOf("<turn|>", "<eos>")
    },

    /** No template — raw completion. Useful for isolating a template bug. */
    RAW("Raw completion") {
        override fun format(systemPrompt: String, userPrompt: String): String = userPrompt
        override fun stopTokens() = emptyList<String>()
    };

    abstract fun format(systemPrompt: String, userPrompt: String): String

    /** Markers to strip from streamed output, and to stop generating on. */
    abstract fun stopTokens(): List<String>

    companion object {
        /** Best guess from the .pte filename, so the default is usually right. */
        fun guessFrom(modelPath: String): PromptFormat {
            val name = modelPath.lowercase()
            return when {
                name.contains("llama") -> LLAMA3
                name.contains("qwen") -> CHATML
                // Order matters: "gemma4" must be checked before "gemma".
                name.contains("gemma4") || name.contains("gemma_4") -> GEMMA4
                name.contains("gemma") -> GEMMA
                name.contains("smollm") -> CHATML
                else -> CHATML
            }
        }
    }
}
