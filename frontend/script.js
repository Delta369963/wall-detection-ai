const uploadBtn =
    document.getElementById("uploadBtn");

const imageInput =
    document.getElementById("imageInput");

const originalImage =
    document.getElementById("originalImage");

const resultImage =
    document.getElementById("resultImage");

uploadBtn.addEventListener(
    "click",
    async () => {

        const file =
            imageInput.files[0];

        if (!file) {

            alert(
                "Please upload an image."
            );

            return;
        }

        // -----------------------------------
        // Original preview
        // -----------------------------------

        originalImage.src =
            URL.createObjectURL(file);

        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );

        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            if (!response.ok) {

                throw new Error(
                    "Prediction failed."
                );
            }

            // -----------------------------------
            // Convert backend image
            // -----------------------------------

            const blob =
                await response.blob();

            const imageUrl =
                URL.createObjectURL(blob);

            resultImage.src =
                imageUrl;

        }

        catch (error) {

            console.error(error);

            alert(
                "Backend prediction failed."
            );
        }
    }
);