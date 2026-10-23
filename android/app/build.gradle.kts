plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// DSS address. Override at build time with -PappUrl=https://your-server (or the GitHub "Run workflow" box).
val appUrl: String = (project.findProperty("appUrl") as String?)?.takeIf { it.isNotBlank() }
    ?: "https://dssrwh-mdqvxegbsnd2hkc5vpp8mf.streamlit.app"
val runNumber: Int = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 1
val keystoreFile: String? = System.getenv("KEYSTORE_FILE")

android {
    namespace = "org.icarneh.geopond"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.icarneh.geopond"
        minSdk = 26
        targetSdk = 34
        versionCode = runNumber
        versionName = "2.2.$runNumber"
        buildConfigField("String", "APP_URL", "\"$appUrl\"")
    }

    signingConfigs {
        create("release") {
            if (keystoreFile != null) {
                storeFile = file(keystoreFile)
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                keyAlias = System.getenv("KEY_ALIAS")
                keyPassword = System.getenv("KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (keystoreFile != null) signingConfig = signingConfigs.getByName("release")
        }
    }

    buildFeatures { buildConfig = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation("androidx.webkit:webkit:1.11.0")
}
