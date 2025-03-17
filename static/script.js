const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const captureBtn = document.getElementById("captureBtn");
const resultText = document.getElementById("result");

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

async function captureAndCompare() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received Data:", data); // Debugging

        // Show result message
        resultText.style.fontSize = "20px";
        resultText.style.fontWeight = "bold";
        resultText.style.textAlign = "center";
        resultText.style.padding = "10px";

        if (data.match) {
            resultText.innerHTML = "✅ Face Matched! Redirecting...";
            resultText.style.color = "green";

            // Redirect to dashboard after 2 seconds
            setTimeout(() => {
                window.location.href = "https://samridhhgov.netlify.app/dashboard.html";
            }, 2000);
        } else {
            resultText.innerHTML = "❌ Face Not Matched";
            resultText.style.color = "red";
        }

        // Draw bounding box on the detected face
        if (data.box) {
            const [x, y, x2, y2] = data.box;
            ctx.strokeStyle = data.match ? "green" : "red";
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, x2 - x, y2 - y);
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.innerHTML = "⚠ Error Processing Request";
        resultText.style.color = "orange";
    });
}

captureBtn.addEventListener("click", captureAndCompare);
startCamera();
