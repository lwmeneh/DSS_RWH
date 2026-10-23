# GeoPond-DSS Android app

A native Android shell (Kotlin WebView) around the GeoPond-DSS Streamlit server.
The GIS analysis runs on the server, so the phone needs internet.

## Get the APK (no Android Studio needed)

1. Copy the `android/` folder and `.github/workflows/build-apk.yml` into your GitHub repo
   (same repo as `app.py`, keeping the paths) and push.
2. GitHub → **Actions** → *Build GeoPond-DSS Android APK* runs automatically
   (or click **Run workflow**; you can type a different server address there).
3. After about 5–8 minutes:
   - **Releases** (right side of the repo page) → `GeoPond-DSS-v2.2.N.apk`: a direct link to share with field staff.
   - Or **Actions** → the run → *Artifacts* → `GeoPond-DSS-APK`.
4. On the phone: open the APK → allow **Install unknown apps** → Install.

## Updatable builds (recommended before wide distribution)

Without a signing key, GitHub builds a *debug* APK with a new key each time, so users must
uninstall the old version before installing a new one. Create one permanent key, once:

```bash
keytool -genkeypair -v -keystore geopond-release.jks -alias geopond \
  -keyalg RSA -keysize 2048 -validity 10000
base64 -w0 geopond-release.jks > keystore.txt      # macOS: base64 -i geopond-release.jks
```

GitHub repo → Settings → Secrets and variables → Actions → add:

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | contents of `keystore.txt` |
| `ANDROID_KEYSTORE_PASSWORD` | keystore password |
| `ANDROID_KEY_ALIAS` | `geopond` |
| `ANDROID_KEY_PASSWORD` | key password |

Keep `geopond-release.jks` and the passwords safe: losing them means users must reinstall,
and Play Store updates become impossible.

## Features

- Loads the DSS in its compact phone layout (`?embed=true&app=android`); always light theme
- GPS: location permission bridged to the map's ◎ button; prompts if device location is off
- **GPS fix check** (toolbar ◎ icon): live lat/lon and accuracy with a quality rating; copy coordinates
- Boundary upload: GeoJSON / KML / GPKG / zipped SHP via the phone's file picker
- Downloads: HTML report, CSV and GeoJSON saved to `Downloads/GeoPond`, with a notification
- Google Maps "Navigate to site" opens the Maps app; other external links open in the browser
- Offline / server-error screen with Retry; double-back to exit; screen kept awake while open
- Menu: reload, start page, share app link, open in browser, **server address** (switch to an
  institute server without rebuilding), about

## Change the default server

Run the workflow with a new address, or edit `appUrl` in `app/build.gradle.kts`. Users can also
change it on the phone: menu → *Server address*.

## Build locally instead

Open the `android/` folder in Android Studio (Koala or newer) → *Build → Build APK(s)*.
