#!/usr/bin/env bash
# Shared Android toolchain env. Source this, don't execute it:
#   source scripts/android_env.sh
#
# Both of these are installed by Homebrew but not exported by default, which
# is the usual reason a Gradle build works in one shell and not another.
export ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
export JAVA_HOME="${JAVA_HOME:-$(brew --prefix openjdk@21 2>/dev/null)}"
export PATH="$ANDROID_HOME/platform-tools:$JAVA_HOME/bin:$PATH"
