$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:5000/")
$listener.Start()
Write-Host "Backend API Server running on http://localhost:5000/"

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        
        # Enable CORS headers
        $response.AddHeader("Access-Control-Allow-Origin", "*")
        $response.AddHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        $response.AddHeader("Access-Control-Allow-Headers", "Content-Type")

        if ($request.HttpMethod -eq "OPTIONS") {
            $response.StatusCode = 200
            $response.OutputStream.Close()
            continue
        }

        $path = $request.Url.LocalPath
        $response.ContentType = "application/json"

        if ($path -eq "/api/health") {
            $json = '{"status":"ONLINE","version":"1.0.0","architecture":"Modular ML Framework","modules":5}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($path -eq "/api/predict/diet") {
            $json = '{"algorithm":"Random Forest Classifier (Paper 89.2% Acc)","dietScore":89.2,"ratingCategory":"Optimal","status":"SUCCESS"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($path -eq "/api/predict/disease") {
            $json = '{"algorithm":"Multiclass Logistic Regression (Paper 86.7% Acc)","f1Score":0.84,"primaryDiagnosis":"Flu / Viral Influenza","status":"SUCCESS"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($path -eq "/api/predict/ecg") {
            $json = '{"algorithm":"Weighted Soft-Voting Ensemble (Paper 93.5% Acc, 0.96 AUC)","modelAccuracy":93.5,"status":"SUCCESS"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($path -eq "/api/predict/fitness") {
            $json = '{"algorithm":"Linear Regression (Paper R² 0.82)","r2Score":0.82,"fitnessScore":82.5,"status":"SUCCESS"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        elseif ($path -eq "/api/predict/risk") {
            $json = '{"algorithm":"XGBoost Classifier (Paper 91.3% Acc)","riskScore":42,"riskLevel":"Moderate","status":"SUCCESS"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        else {
            $response.StatusCode = 404
            $json = '{"error":"Endpoint not found"}'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        $response.OutputStream.Close()
    }
} finally {
    $listener.Stop()
}
