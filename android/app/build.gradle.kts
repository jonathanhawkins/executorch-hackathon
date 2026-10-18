plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "ai.executorch.hackathon"
    compileSdk = 36

    defaultConfig {
        applicationId = "ai.executorch.hackathon"
        // The AAR itself allows 23; 28 keeps us on devices with a sane NNAPI
        // and 64-bit-only userspace, which is everything in the three tracks.
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"

        ndk {
            // The AAR ships arm64-v8a (devices) and x86_64 (emulator) only.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation(libs.executorch.android)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    debugImplementation(libs.compose.ui.tooling)
}
