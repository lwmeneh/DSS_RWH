package org.icarneh.geopond

import android.Manifest
import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Looper
import android.provider.MediaStore
import android.provider.Settings
import android.util.Base64
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.GeolocationPermissions
import android.webkit.JavascriptInterface
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.webkit.WebSettingsCompat
import androidx.webkit.WebViewFeature
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import java.io.File
import java.util.Locale

/**
 * GeoPond-DSS Android shell.
 *
 * Loads the Streamlit DSS in a WebView and adds what a plain browser tab lacks in the field:
 * GPS permission bridging, boundary-file upload, report / CSV / GeoJSON downloads, external
 * navigation links, an offline screen, a native GPS-fix check and a configurable server address.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var errorView: View
    private lateinit var errorTitle: TextView
    private lateinit var errorMsg: TextView

    private val prefs by lazy { getSharedPreferences("geopond", Context.MODE_PRIVATE) }
    private var lastBackPress = 0L
    private var pageLoaded = false

    // Pending WebView requests waiting for an Android permission / picker result
    private var geoOrigin: String? = null
    private var geoCallback: GeolocationPermissions.Callback? = null
    private var pendingGpsCheck = false
    private var filePathCallback: ValueCallback<Array<Uri>>? = null

    private val locationPermission =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val granted = result[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                result[Manifest.permission.ACCESS_COARSE_LOCATION] == true
            geoCallback?.invoke(geoOrigin, granted, false)
            geoCallback = null
            if (!granted) toast("Location permission is needed for the GPS button on the map.")
            if (pendingGpsCheck) {
                pendingGpsCheck = false
                if (granted) showGpsFix()
            }
        }

    private val fileChooser =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            val uris = WebChromeClient.FileChooserParams.parseResult(res.resultCode, res.data)
            filePathCallback?.onReceiveValue(uris)
            filePathCallback = null
        }

    // ------------------------------------------------------------------ lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Keep the screen awake while the app is open (field mapping / GPS walking).
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)

        webView = findViewById(R.id.webview)
        progress = findViewById(R.id.progress)
        errorView = findViewById(R.id.error_view)
        errorTitle = findViewById(R.id.error_title)
        errorMsg = findViewById(R.id.error_msg)
        findViewById<MaterialButton>(R.id.retry_button).setOnClickListener { loadHome() }

        configureWebView()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    errorView.visibility == View.VISIBLE -> finish()
                    webView.canGoBack() -> webView.goBack()
                    System.currentTimeMillis() - lastBackPress < 2000 -> finish()
                    else -> {
                        lastBackPress = System.currentTimeMillis()
                        toast("Press back again to exit GeoPond-DSS")
                    }
                }
            }
        })

        if (savedInstanceState != null) webView.restoreState(savedInstanceState) else loadHome()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
    }

    override fun onResume() {
        super.onResume()
        webView.onResume()
    }

    override fun onPause() {
        webView.onPause()
        CookieManager.getInstance().flush()
        super.onPause()
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ WebView setup

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        with(webView.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            setGeolocationEnabled(true)
            mediaPlaybackRequiresUserGesture = true
            loadWithOverviewMode = true
            useWideViewPort = true
            builtInZoomControls = false
            displayZoomControls = false
            setSupportZoom(false)
            setSupportMultipleWindows(false)
            javaScriptCanOpenWindowsAutomatically = true
            cacheMode = WebSettings.LOAD_DEFAULT
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            textZoom = 100
            // Lets the DSS recognise the app and switch to its compact mobile layout.
            userAgentString = "$userAgentString GeoPondAndroid/${BuildConfig.VERSION_NAME}"
        }
        // Never invert colours in phone dark mode — the DSS uses its own contrast palette.
        if (WebViewFeature.isFeatureSupported(WebViewFeature.ALGORITHMIC_DARKENING)) {
            WebSettingsCompat.setAlgorithmicDarkeningAllowed(webView.settings, false)
        }
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true)

        webView.addJavascriptInterface(DownloadBridge(), "GeoPondBridge")
        webView.webViewClient = DssWebViewClient()
        webView.webChromeClient = DssChromeClient()
        webView.setDownloadListener { url, userAgent, contentDisposition, mimeType, _ ->
            handleDownload(url, userAgent, contentDisposition, mimeType)
        }
    }

    private fun baseUrl(): String =
        prefs.getString("server_url", null)?.takeIf { it.isNotBlank() } ?: BuildConfig.APP_URL

    /** DSS URL with the parameters that select the embedded, compact app layout. */
    private fun appUrl(): String {
        val base = baseUrl().trimEnd('/')
        val sep = if (base.contains('?')) "&" else "?"
        return "$base${sep}embed=true&app=android"
    }

    private fun appHost(): String = Uri.parse(baseUrl()).host.orEmpty()

    private fun loadHome() {
        if (!isOnline()) {
            showError(getString(R.string.offline_title), getString(R.string.offline_msg))
            return
        }
        errorView.visibility = View.GONE
        webView.visibility = View.VISIBLE
        pageLoaded = false
        webView.loadUrl(appUrl())
    }

    private fun showError(title: String, message: String) {
        errorTitle.text = title
        errorMsg.text = message
        errorView.visibility = View.VISIBLE
        progress.visibility = View.GONE
    }

    private fun isOnline(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    /** Pages that stay inside the app; everything else (e.g. Google Maps navigation) opens externally. */
    private fun isInternalHost(host: String?): Boolean {
        if (host.isNullOrEmpty()) return true
        val h = host.lowercase(Locale.ROOT)
        return h == appHost().lowercase(Locale.ROOT) ||
            h.endsWith(".streamlit.app") || h == "streamlit.app" ||
            h.endsWith("streamlit.io")
    }

    private inner class DssWebViewClient : WebViewClient() {
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
            if (!request.isForMainFrame) return false
            val uri = request.url
            val scheme = uri.scheme?.lowercase(Locale.ROOT)
            if ((scheme == "http" || scheme == "https") && isInternalHost(uri.host)) return false
            openExternal(uri)
            return true
        }

        override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
            progress.visibility = View.VISIBLE
        }

        override fun onPageFinished(view: WebView, url: String?) {
            pageLoaded = true
            progress.visibility = View.GONE
        }

        override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
            if (request.isForMainFrame) {
                showError(
                    getString(R.string.error_title),
                    "${error.description}\n\nCheck the connection or the server address (menu → Server address)."
                )
            }
        }
    }

    private inner class DssChromeClient : WebChromeClient() {
        override fun onProgressChanged(view: WebView, newProgress: Int) {
            progress.setProgressCompat(newProgress, true)
            progress.visibility = if (newProgress < 100) View.VISIBLE else View.GONE
        }

        override fun onGeolocationPermissionsShowPrompt(
            origin: String?, callback: GeolocationPermissions.Callback?
        ) {
            if (hasLocationPermission()) {
                callback?.invoke(origin, true, false)
                warnIfLocationOff()
            } else {
                geoOrigin = origin
                geoCallback = callback
                requestLocationPermission()
            }
        }

        override fun onShowFileChooser(
            view: WebView,
            callback: ValueCallback<Array<Uri>>,
            params: FileChooserParams
        ): Boolean {
            filePathCallback?.onReceiveValue(null)
            filePathCallback = callback
            // GeoJSON / KML / GPKG / zip have no reliable MIME types on Android → allow any file.
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "*/*"
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, params.mode == FileChooserParams.MODE_OPEN_MULTIPLE)
            }
            return try {
                fileChooser.launch(Intent.createChooser(intent, "Select field boundary file"))
                true
            } catch (e: Exception) {
                filePathCallback = null
                toast("No file picker available")
                false
            }
        }
    }

    // ------------------------------------------------------------------ location

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestLocationPermission() {
        locationPermission.launch(
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        )
    }

    private fun locationEnabled(): Boolean {
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        return lm.isProviderEnabled(LocationManager.GPS_PROVIDER) ||
            lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
    }

    private fun warnIfLocationOff() {
        if (!locationEnabled()) {
            AlertDialog.Builder(this)
                .setTitle("Location is off")
                .setMessage("Turn on device location (GPS) to use the map's GPS button.")
                .setPositiveButton("Open settings") { _, _ ->
                    startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
                }
                .setNegativeButton("Later", null)
                .show()
        }
    }

    /** Live satellite fix with accuracy, so field staff know when GPS is good enough to map. */
    @SuppressLint("MissingPermission")
    private fun showGpsFix() {
        if (!hasLocationPermission()) {
            pendingGpsCheck = true
            requestLocationPermission()
            return
        }
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if (!lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            warnIfLocationOff()
            return
        }
        var last: Location? = null
        val fixView = TextView(this).apply {
            setPadding(56, 32, 56, 8)
            textSize = 16f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            text = "Waiting for satellite fix…\nStand in the open, away from buildings and trees."
        }
        val listener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                last = location
                val acc = location.accuracy
                val quality = when {
                    acc <= 5 -> "Excellent — good for boundary mapping"
                    acc <= 10 -> "Good"
                    acc <= 20 -> "Fair — wait a little longer"
                    else -> "Poor — move to an open spot"
                }
                fixView.text = String.format(
                    Locale.US,
                    "Latitude   %.6f\nLongitude  %.6f\nAccuracy   ± %.1f m\nAltitude   %.0f m\n\n%s",
                    location.latitude, location.longitude, acc, location.altitude, quality
                )
            }

            override fun onProviderEnabled(provider: String) {}
            override fun onProviderDisabled(provider: String) {}

            @Deprecated("Deprecated in Java")
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        }
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, listener, Looper.getMainLooper())

        AlertDialog.Builder(this)
            .setTitle("GPS fix check")
            .setView(fixView)
            .setPositiveButton("Copy coordinates") { _, _ ->
                last?.let {
                    val coords = String.format(Locale.US, "%.6f, %.6f", it.latitude, it.longitude)
                    val cb = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    cb.setPrimaryClip(ClipData.newPlainText("GPS", coords))
                    toast("Copied $coords — paste into 'Go to coordinates'")
                } ?: toast("No fix yet")
            }
            .setNegativeButton("Close", null)
            .setOnDismissListener { lm.removeUpdates(listener) }
            .show()
    }

    // ------------------------------------------------------------------ downloads

    private fun handleDownload(url: String, userAgent: String?, contentDisposition: String?, mimeType: String?) {
        val fileName = URLUtil.guessFileName(url, contentDisposition, mimeType)
        when {
            url.startsWith("blob:") -> fetchBlob(url, fileName, mimeType ?: "application/octet-stream")
            url.startsWith("data:") -> {
                val ok = saveBytes(decodeDataUrl(url), fileName, mimeType ?: "application/octet-stream")
                toast(if (ok) "Saved to Downloads/GeoPond: $fileName" else "Could not save $fileName")
            }
            else -> {
                try {
                    val request = DownloadManager.Request(Uri.parse(url)).apply {
                        setMimeType(mimeType)
                        CookieManager.getInstance().getCookie(url)?.let { addRequestHeader("Cookie", it) }
                        userAgent?.let { addRequestHeader("User-Agent", it) }
                        setTitle(fileName)
                        setDescription("GeoPond-DSS download")
                        setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, "GeoPond/$fileName")
                        } else {
                            setDestinationInExternalFilesDir(
                                this@MainActivity, Environment.DIRECTORY_DOWNLOADS, fileName
                            )
                        }
                    }
                    (getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager).enqueue(request)
                    toast("Downloading $fileName…")
                } catch (e: Exception) {
                    toast("Download failed: ${e.message}")
                }
            }
        }
    }

    /** blob: URLs only exist inside the page, so read them in JavaScript and hand the bytes to Kotlin. */
    private fun fetchBlob(url: String, fileName: String, mime: String) {
        val js = """
            (function() {
              var x = new XMLHttpRequest();
              x.open('GET', ${jsString(url)}, true);
              x.responseType = 'blob';
              x.onload = function() {
                var r = new FileReader();
                r.onloadend = function() { GeoPondBridge.saveBase64(r.result, ${jsString(fileName)}, ${jsString(mime)}); };
                r.readAsDataURL(x.response);
              };
              x.onerror = function() { GeoPondBridge.saveBase64('', ${jsString(fileName)}, ${jsString(mime)}); };
              x.send();
            })();
        """.trimIndent()
        webView.evaluateJavascript(js, null)
    }

    private fun jsString(s: String): String =
        "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"

    private fun decodeDataUrl(dataUrl: String): ByteArray {
        val payload = dataUrl.substringAfter(",", "")
        return if (dataUrl.substringBefore(",").contains(";base64")) Base64.decode(payload, Base64.DEFAULT)
        else Uri.decode(payload).toByteArray()
    }

    private fun saveBytes(bytes: ByteArray, fileName: String, mime: String): Boolean {
        if (bytes.isEmpty()) return false
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, mime)
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/GeoPond")
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return false
                contentResolver.openOutputStream(uri)?.use { it.write(bytes) } ?: return false
            } else {
                val dir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS) ?: filesDir
                File(dir, fileName).writeBytes(bytes)
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    private inner class DownloadBridge {
        @JavascriptInterface
        fun saveBase64(dataUrl: String, fileName: String, mime: String) {
            val ok = dataUrl.isNotEmpty() && saveBytes(decodeDataUrl(dataUrl), fileName, mime)
            runOnUiThread {
                toast(if (ok) "Saved to Downloads/GeoPond: $fileName" else "Could not save $fileName")
            }
        }
    }

    // ------------------------------------------------------------------ menu

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.action_reload -> if (pageLoaded && errorView.visibility != View.VISIBLE) webView.reload() else loadHome()
            R.id.action_home -> loadHome()
            R.id.action_gps -> showGpsFix()
            R.id.action_browser -> openExternal(Uri.parse(baseUrl()))
            R.id.action_share -> startActivity(
                Intent.createChooser(
                    Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_SUBJECT, "GeoPond-DSS")
                        putExtra(
                            Intent.EXTRA_TEXT,
                            "GeoPond-DSS — farm pond & Jalkund planning (ICAR NEH): ${baseUrl()}"
                        )
                    }, "Share GeoPond-DSS"
                )
            )
            R.id.action_server -> editServerUrl()
            R.id.action_about -> showAbout()
            else -> return super.onOptionsItemSelected(item)
        }
        return true
    }

    private fun editServerUrl() {
        val input = EditText(this).apply {
            setText(baseUrl())
            setSingleLine()
            setSelection(text.length)
        }
        val box = LinearLayout(this).apply {
            setPadding(56, 16, 56, 0)
            addView(input, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ))
        }
        AlertDialog.Builder(this)
            .setTitle("DSS server address")
            .setMessage("Use this if the DSS moves to an institute server.")
            .setView(box)
            .setPositiveButton("Save") { _, _ ->
                val url = input.text.toString().trim()
                if (url.startsWith("https://")) {
                    prefs.edit().putString("server_url", url).apply()
                    loadHome()
                } else {
                    toast("Address must start with https://")
                }
            }
            .setNeutralButton("Reset to default") { _, _ ->
                prefs.edit().remove("server_url").apply()
                loadHome()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showAbout() {
        AlertDialog.Builder(this)
            .setTitle("GeoPond-DSS ${BuildConfig.VERSION_NAME}")
            .setMessage(
                "Geospatial Pond Siting and Preliminary Design Decision Support System\n\n" +
                    "ICAR Research Complex for NEH Region, Umiam, Meghalaya\n\n" +
                    "Server: ${baseUrl()}\n\n" +
                    "GIS screening and preliminary planning only. Final siting and design require " +
                    "field survey and engineering verification."
            )
            .setPositiveButton("OK", null)
            .show()
    }

    private fun openExternal(uri: Uri) {
        try {
            startActivity(Intent(Intent.ACTION_VIEW, uri))
        } catch (e: Exception) {
            toast("No app found to open this link")
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
}
