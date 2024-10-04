#%%%
import layoutparser as lp
import fitz  # PyMuPDF
from PIL import Image
import pyocr
import pyocr.builders
import io
import os

# pyocrのツールを取得
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    exit(1)
tool = tools[0]

# PDFファイルのパス
pdf_path = "Attention_is_All_You_Need.pdf"
document = fitz.open(pdf_path)

# 画像を保存するディレクトリの作成
output_dir = os.path.splitext(pdf_path)[0]
os.makedirs(output_dir, exist_ok=True)

# テキストファイルのパス
output_text_file = os.path.join(output_dir, "extracted_text.txt")

# レイアウトモデルのロード
model = lp.Detectron2LayoutModel(
    config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# 全文を格納するリスト
full_text = []

# 各ページを処理
for page_num in range(len(document)):
    page = document.load_page(page_num)
    # 解像度を300 DPIに設定してPixMapを生成
    zoom_x = 300 / 72
    zoom_y = 300 / 72
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))  # PIL Imageに変換

    # ページのレイアウトを検出
    layout = model.detect(img)

    # テキストと画像ブロックを分離
    text_blocks = []
    image_blocks = []

    for block in layout:
        if block.type == 'Text':
            text_blocks.append(block)
        elif block.type == 'Figure':
            image_blocks.append(block)

    # 結果を表示
    print(f"Page {page_num + 1}")
    print("Text Blocks:")
    for block in text_blocks:
        print(block)
    print("Image Blocks:")
    for block in image_blocks:
        print(block)

    # 画像ブロックを保存し、ファイル名を挿入
    for i, block in enumerate(image_blocks):
        x1, y1, x2, y2 = map(int, block.coordinates)
        cropped_img = img.crop((x1, y1, x2, y2))
        image_filename = f"page_{page_num + 1}_image_{i + 1}.png"
        cropped_img.save(os.path.join(output_dir, image_filename))
        # 画像の位置にファイル名を挿入
        full_text.append(f"[Image: {image_filename}]")

    # テキストブロックを抽出し、元の位置に挿入
    for block in text_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        cropped_img = img.crop((x1, y1, x2, y2))
        text = tool.image_to_string(
            cropped_img,
            lang="eng",
            builder=pyocr.builders.TextBuilder()
        )
        full_text.append(text)

# 全文をテキストファイルに出力
with open(output_text_file, "w", encoding="utf-8") as f:
        # リストを改行で結合
        f.write("\n".join(full_text))

print("処理が完了しました。")
#%%
