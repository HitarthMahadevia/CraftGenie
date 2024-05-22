import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the application window
app = tk.Tk()
app.geometry("532x622")
app.title("CraftGenie")

# Set appearance mode to dark
ctk.set_appearance_mode("dark")

# Create label for the prompt message
prompt_message = ctk.CTkLabel(app, text="Spark visuals with words...", height=40, width=512, text_color="black")
prompt_message.configure(font=("Arial", 16))
prompt_message.place(x=10, y=10)

# Create entry field
prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white")
prompt.configure(font=("Arial", 20))
prompt.place(x=10, y=50)

# Create label for displaying images
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=133)

# Create label for status messages
status_label = ctk.CTkLabel(app, text="", height=40, width=512, text_color="black")
status_label.configure(font=("Arial", 16))
status_label.place(x=10, y=129)

# Load the stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=auth_token
)
pipe.to(device)

def generate():
    """
    Generate an image based on the text input in the entry field.
    """
    # Update the status label to show the message
    status_label.configure(text="🎨✨ Hang tight! CraftGenie is conjuring your masterpiece...")
    app.update_idletasks()

    try:
        # Generate the image
        with autocast(device):
            result = pipe(prompt.get(), guidance_scale=8.5)

        # Inspect the output keys
        if 'images' in result:
            image = result['images'][0]
        else:
            status_label.configure(text="❌ Error: Unable to generate image.")
            return

        # Save the generated image
        image.save("generated.png")

        # Convert the generated image to PhotoImage format
        img = ImageTk.PhotoImage(image)

        # Keep a reference to the image to prevent it from being garbage collected
        lmain.image = img
        
        # Update the label to display the generated image
        lmain.configure(image=img)

    except Exception as e:
        status_label.configure(text=f"❌ Error: {str(e)}")

    # Clear the status message
    status_label.configure(text="")

# Create a button for generating images
trigger = ctk.CTkButton(
    app,
    height=40,
    width=120,
    text="Generate",
    font=("Arial", 16),
    text_color="white",
    fg_color="blue",
    command=generate
)
trigger.place(x=206, y=100)

# Start the application event loop
app.mainloop()
