"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License

https://github.com/clovaai/donut
"""
import gradio as gr

from donut import DonutModel


def demo_process(input_img):
    global pretrained_model, task_prompt, task_name
    # input_img = Image.fromarray(input_img)
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)[
        "predictions"
    ][0]
    return output


# task_prompt = f"<s_cord-v2>"
task_prompt = f"<s_receipts>"

device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'

pretrained_model: DonutModel = DonutModel.from_pretrained(
    "result/train_receipts_local/20230511_144806", local_files_only=True
)
pretrained_model.to(device)
pretrained_model.eval()

demo = gr.Interface(
    fn=demo_process,
    inputs=gr.inputs.Image(type="pil"),
    outputs="json",
    title=f"Donut 🍩 demonstration for `cord-v2` task",
    description="""This model is trained with 800 Indonesian receipt images of CORD dataset. <br>
Demonstrations for other types of documents/tasks are available at https://github.com/clovaai/donut <br>
More CORD receipt images are available at https://huggingface.co/datasets/naver-clova-ix/cord-v2

More details are available at:
- Paper: https://arxiv.org/abs/2111.15664
- GitHub: https://github.com/clovaai/donut""",
    examples=[
        ["sample_image_cord_test_receipt_00004.png"],
        ["sample_image_cord_test_receipt_00012.png"],
    ],
    cache_examples=False,
)

demo.launch(server_name="0.0.0.0")
