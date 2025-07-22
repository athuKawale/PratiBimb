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


// Main swap logic: calls multifaceswap API and updates output image
async function swapFaces() {
    const sourceInput = document.getElementById('source-images');
    const targetInput = document.getElementById('target-image');
    const outputPreview = document.getElementById('outputPreview');

    // Basic client-side validation
    if (sourceInput.files.length === 0) {
        alert("Please select at least one source image.");
        return;
    }
    if (targetInput.files.length === 0) {
        alert("Please select a target image.");
        return;
    }

    const formData = new FormData();
    // Append all source images
    for (let i = 0; i < sourceInput.files.length; i++) {
        formData.append('source_images', sourceInput.files[i]);
    }
    // Append target image
    formData.append('target_image', targetInput.files[0]);

    try {
        // POST to backend API endpoint
        const response = await fetch('/api/v1/multifaceswap', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            // Try to extract meaningful error message
            let errorText = "Unknown error occurred.";
            try {
                const errorData = await response.json();
                errorText = errorData.error || errorText;
            } catch (_) {}
            alert('Face swap failed: ' + errorText);
            return;
        }

        // Get returned swapped image as a blob
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Set output image preview
        outputPreview.src = imageUrl;
    } catch (error) {
        alert('Failed to swap faces: ' + error.message);
    }
}
