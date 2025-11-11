from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image


class PromptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 prompt_col: str = "quesion",
                 image_col:str = "image_path",
                 max_size = 512,
                 add_options=True):
        
        self.df = df
        self.prompt_col = prompt_col
        self.image_col = image_col
        self.max_size = max_size
        self.add_options = add_options

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"])
        scale = 1.0
        if max(image.size) > self.max_size:
            image,scale = self.re_scale(image)
        
        new_w, new_h = image.size

        if self.add_options:
            question =  row["question"] +  f" Options: {row['options']}"
        else:
            question = row["question"]
        
        return {
            "index": int(row["index"]),  
            "question": question, 
            "options": row["options"],                # original df index
            "image_path":row["image_path"],
            "dataset":row["dataset"],
            "class_label":row["class_label"],

           
            "image": image,
            "image_scale":scale,
            "scaled_width":new_w,
            "scaled_height":new_h,
        }
    def re_scale(self,image):
            orig_w, orig_h = image.size
            scale = self.max_size / max(image.size)
            new_w_, new_h_ = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            #new_w, new_h = image.size
            return image,scale
        
def prompt_collate(batch):
    # keep as list[str] so vLLM can take it directly
    return batch


def create_template(item):
    conversation = {
            "prompt": f"USER: <image>\n{item['question']}\nASSISTANT:",
            "multi_modal_data": {"image": item['image']},
        }
    return conversation

