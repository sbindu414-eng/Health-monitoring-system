$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:8000/")
$listener.Start()
Write-Host "Server running on http://localhost:8000/"

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        $path = $request.Url.LocalPath
        if ($path -eq "/") { $path = "/index.html" }
        
        $baseDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
        $localPath = Join-Path $baseDir $path.TrimStart('/')
        
        if (Test-Path $localPath -PathType Leaf) {
            $bytes = [System.IO.File]::ReadAllBytes($localPath)
            if ($localPath.EndsWith(".html")) { $response.ContentType = "text/html; charset=utf-8" }
            elseif ($localPath.EndsWith(".css")) { $response.ContentType = "text/css" }
            elseif ($localPath.EndsWith(".js")) { $response.ContentType = "application/javascript" }
            elseif ($localPath.EndsWith(".json")) { $response.ContentType = "application/json" }
            elseif ($localPath.EndsWith(".svg")) { $response.ContentType = "image/svg+xml" }
            
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $response.StatusCode = 404
        }
        $response.OutputStream.Close()
    }
} finally {
    $listener.Stop()
}
