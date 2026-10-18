package ai.executorch.hackathon

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.material3.MenuAnchorType

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    ChatScreen(Modifier.padding(padding))
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(modifier: Modifier = Modifier, vm: ChatViewModel = viewModel()) {
    val scrollState = rememberScrollState()
    var formatMenuOpen by remember { mutableStateOf(false) }

    LaunchedEffect(vm.output) { scrollState.animateScrollTo(scrollState.maxValue) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .imePadding()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("ExecuTorch on-device", style = MaterialTheme.typography.titleLarge)

        OutlinedTextField(
            value = vm.modelPath,
            onValueChange = vm::onModelPathChange,
            label = { Text("Model .pte") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )

        OutlinedTextField(
            value = vm.tokenizerPath,
            onValueChange = { vm.tokenizerPath = it },
            label = { Text("Tokenizer") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                onClick = vm::loadModel,
                enabled = !vm.isBusy,
            ) { Text(if (vm.isLoaded) "Reload" else "Load model") }

            ExposedDropdownMenuBox(
                expanded = formatMenuOpen,
                onExpandedChange = { formatMenuOpen = it },
                modifier = Modifier.weight(1f),
            ) {
                OutlinedTextField(
                    value = vm.format.label,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Chat template") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(formatMenuOpen) },
                    modifier = Modifier
                        .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                        .fillMaxWidth(),
                )
                ExposedDropdownMenu(
                    expanded = formatMenuOpen,
                    onDismissRequest = { formatMenuOpen = false },
                ) {
                    PromptFormat.entries.forEach { fmt ->
                        DropdownMenuItem(
                            text = { Text(fmt.label) },
                            onClick = {
                                vm.format = fmt
                                formatMenuOpen = false
                            },
                        )
                    }
                }
            }
        }

        if (vm.status.isNotBlank()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    vm.status,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Card(modifier = Modifier.weight(1f).fillMaxWidth()) {
            Text(
                text = vm.output.ifBlank { "Load a model, then ask it something." },
                modifier = Modifier
                    .verticalScroll(scrollState)
                    .padding(12.dp),
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodyMedium,
            )
        }

        OutlinedTextField(
            value = vm.prompt,
            onValueChange = { vm.prompt = it },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = vm::generate,
                enabled = vm.isLoaded && !vm.isBusy && vm.prompt.isNotBlank(),
                modifier = Modifier.weight(1f),
            ) {
                if (vm.isGenerating) {
                    CircularProgressIndicator(modifier = Modifier.padding(end = 8.dp))
                }
                Text(if (vm.isGenerating) "Generating…" else "Generate")
            }

            OutlinedButton(onClick = vm::stop, enabled = vm.isGenerating) { Text("Stop") }
            OutlinedButton(onClick = vm::clear, enabled = !vm.isBusy) { Text("Clear") }
        }
    }
}
