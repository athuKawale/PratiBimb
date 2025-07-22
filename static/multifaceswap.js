// Source images: preview thumbnails
document.getElementById('source-images').addEventListener('change', function (event) {
    const files = event.target.files;
    const container = document.getElementById('sourcePreviewContainer');
    container.innerHTML = ""; // Clear previous previews
    for (let i = 0; i < files.length; i++) {
        const img = document.createElement('img');
        img.className = "border rounded bg-white";
        img.style.width = "110px";
        img.style.height = "110px";
        img.style.objectFit = "contain";
        img.src = URL.createObjectURL(files[i]);
        img.alt = "Source " + (i + 1);
        container.appendChild(img);
    }
});

// Target image: show preview and trigger face detection
document.getElementById('target-image').addEventListener('change', function (event) {
    const file = event.target.files[0];
    const targetPreview = document.getElementById('targetPreview');
    const targetPreviewBox = document.getElementById('targetPreviewBox');
    const detectedFacesSection = document.getElementById('detectedFacesSection');

    if (file) {
        const imageURL = URL.createObjectURL(file);
        targetPreview.src = imageURL;
        targetPreviewBox.classList.remove("d-none");

        // Show and populate the detected faces section
        detectedFacesSection.classList.remove("d-none");
        detectFacesInTarget(imageURL); // Call face detection simulation
    } else {
        targetPreview.src = '';
        targetPreviewBox.classList.add("d-none");
        detectedFacesSection.classList.add("d-none"); // Hide if no file
    }
});

// Stub function to simulate face detection
function detectFacesInTarget(imageURL) {
    const container = document.getElementById('detectedFacesContainer');
    container.innerHTML = ""; // Clear previous faces

    // In a real app, you would get face coordinates from an API.
    // Here, we'll just display the same image 3 times as a placeholder.
    for (let i = 0; i < 3; i++) {
        const img = document.createElement('img');
        img.className = "border rounded bg-white";
        img.style.width = "75px";
        img.style.height = "75px";
        img.style.objectFit = "cover"; // Use 'cover' to simulate a cropped face
        img.src = imageURL;
        img.alt = "Detected Face " + (i + 1);
        container.appendChild(img);
    }
}

// Stub for main swap logic
function swapFaces() {
    const targetPreview = document.getElementById('targetPreview');
    const outputPreview = document.getElementById('outputPreview');
    if (targetPreview.src) {
        outputPreview.src = targetPreview.src;
    }
}
