import cv2
from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
import onnxruntime as ort
import numpy as np
import onnxruntime as ort

class ONNXInferencer:
    def __init__(self, model_path):
        self.model_path = model_path
        # 获取输入和输出的名称
        self.session = ort.InferenceSession(
            self.model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"当前推理:{self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        #有capsule
        # self.labels = ['battery', 'blank', 'bottles', 'brick', 'cans', 'capsule', 'carrots', 'china', 'cobble', 'medicine_box', 'medicine_plate', 'ointment', 'potato', 'turnip']
        #无 capsule
        self.labels = ['battery', 'blank', 'bottles', 'brick', 'cans', 'carrots', 'china',
                       'cobble', 'medicine_box', 'medicine_plate', 'ointment', 'potato', 'turnip']

    def load(self) -> None:
        # 加载ONNX模型
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path_or_bytes=self.model_file,providers=providers)
        print(f"当前使用:{self.session.get_providers()}")

    def infer(self, input_data):
        # 此处的input_data应为numpy数组或与模型输入兼容的数据类型
        assert isinstance(
            input_data, np.ndarray), "Input data should be numpy ndarray."
        # 进行推理
        ort_inputs = {self.input_name: input_data}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        # 返回推理结果
        return ort_outputs[0]

    def process_image(self, image: Image.Image):
        """将PIL.Image对象转换为ResNet18可以接受的格式。
        Args:
            image (PIL.Image): 输入的PIL.Image图像。
        Returns:
            np.ndarray: 转换后的图像numpy数组。
        """
        # 定义预处理步骤
        preprocess = transforms.Compose([
            # transforms.Resize((224,224)),  # 缩放图像大小
            # transforms.CenterCrop(224),  # 中心裁剪得到224x224的图像
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),  # 归一化
        ])
        # 应用预处理
        image_tensor = preprocess(image)
        # 转换为适合模型输入的numpy数组
        image_tensor_batch = np.expand_dims(image_tensor, axis=0)
        # 注意：PyTorch模型预期的输入是float32类型
        return image_tensor_batch.astype(np.float32)

    def predict(self, image: Image.Image):
        # 预处理图像
        image = self.process_image(image)
        # 运行ONNX Runtime会话，获取输出
        ort_outs = self.session.run(
            [self.output_name], {self.input_name: image})
        # print(f"ort_outs:{ort_outs}")
        # 使用softmax获得概率分布
        output_probabilities = ort_outs[0]
        softmax_output = np.exp(output_probabilities) / \
            np.sum(np.exp(output_probabilities))
        # 获取概率最高的类别索引
        max_index = np.argmax(softmax_output)
        # 返回概率最高的标签和其置信度
        return self.process_output(softmax_output)

    def process_output(self, probabilities: np.ndarray) -> dict:
        # 转换概率为列表并与标签结合
        # print(*zip(self.labels, *probabilities))
        label_probabilities = list(zip(self.labels, *probabilities))
        # print(label_probabilities)
        # 创建包含标签和对应置信度的字典
        output = [{"label": label, "confidence": float(confidence)}
                  for label, confidence in label_probabilities]
        # 按置信度降序排序预测
        sorted_output = sorted(
            output, key=lambda k: k["confidence"], reverse=True)
        # 包装在"predictions"键下，创建最终的字典
        result = {"predictions": sorted_output}
        return result

if __name__=='__main__':
    model=ONNXInferencer('resnet/resnet34.onnx')
    img=cv2.imread('/home/nano/lobe/data/inner_packaging/1.jpg')
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(0,10):
        outputs = model.predict(pil_image)
        top_prediction = outputs["predictions"][0]
        print(f'outputs:{outputs}\n')
        counter = 0
        for prediction in outputs['predictions']:
            print(f"{prediction['label']}:{prediction['confidence']:.6f}", end='\t\t')
            counter += 1
            if counter % 3 == 0:
                print()
        if counter % 3 != 0:  # 如果最后不是三的倍数，则添加一个换行
            print()
        print(f'top_predict:{top_prediction}\n')