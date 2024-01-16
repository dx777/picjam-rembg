from rembg import remove, new_session
from PIL import Image

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input image",
            default="",
        ),
    ) -> Path:

        session = new_session(model_name='isnet-general-use')
        
        image = Image.open(input_path)
        output = remove(image, session=session, alpha_matting=True, alpha_matting_erode_size=5, post_process_mask=True)
        output_path = f"/tmp/out.png"
        output.save(output_path)

        return Path(output_path)