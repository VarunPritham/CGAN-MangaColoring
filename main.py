from pathlib import Path
import argparse
import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans

import cgan
from cgan import *
class MangaColorization:

    """
    This class is used to colorize the manga image.
    """

    __slots__ = ['image_path', 'image', 'colorized_output']


    def __init__(self, image_path,output_path,checkpoint_file):
        """
        This function is used to initialize the class.
        :param image_path:  input image path
        :param output_path:  output image path
        :param checkpoint_file:  checkpoint file path
        """
        self.image_path = str(image_path)
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)


        gen_model = Generator()
        self.colorized_output = self.test(gen_model, torch.device('cuda'),checkpoint_file)


        colorized_image = self.colorize()

        cv2.imshow('colorized_image', colorized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(str(output_path), colorized_image)

    def test(self, model, device,checkpoint_file):
        """
        This function is used to test the model.
        :param model: pytorch model
        :param device: cpu or gpu
        :param checkpoint_file: checkpoint file path
        :return:
        """
        image = Image.open(self.image_path)
        image_tensor = transforms.ToTensor()(image.resize((512, 512)))

        image_tensor.to(device)
        model.load_state_dict(torch.load(checkpoint_file)['model_state_dict'])
        model.to(device)
        model.eval()
        image_tensor = image_tensor.to(device)
        output = model(image_tensor[:3, :, :].unsqueeze(0))
        output = output.detach().cpu().numpy()
        output = output[0].transpose(1, 2, 0)
        output = output * 255
        output = output.astype('uint8')
        return output


    def apply_gaussian_filter(self, kernel_size, sigma_x):
        """
        This function is used to apply gaussian filter.
        :param kernel_size: filter kernel size
        :param sigma_x: kernel sigma
        :return: blurred image
        """
        image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma_x)
        return image

    def difference_of_gaussian(self, kernel_size):
        """
        This function is used to apply difference of gaussian.
        :param kernel_size:  filter kernel size
        :return: image with enhanced edges
        """
        first_gaussian = self.apply_gaussian_filter(kernel_size, 0)
        second_gaussian = self.apply_gaussian_filter(kernel_size-2, 0)

        return second_gaussian - first_gaussian

    def apply_sobel(self, kernel_size):
        """
        This function is used to apply sobel filter.
        :param kernel_size: filter kernel size
        :return: image with only edges
        """

        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        self.image = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    def apply_prewitt(self):
        """
        This function is used to apply prewitt filter.
        :return: image with only edges
        """

        prewitt_kernel_x = np.array([[-1, 0, 1],
                                     [-1, 0, 1],
                                     [-1, 0, 1]])

        prewitt_kernel_y = np.array([[-1, -1, -1],
                                     [0, 0, 0],
                                     [1, 1, 1]])

        horizontal_edges = cv2.filter2D(self.image, -1, prewitt_kernel_x)
        vertical_edges = cv2.filter2D(self.image, -1, prewitt_kernel_y)

        self.image = cv2.addWeighted(horizontal_edges, 0.5, vertical_edges, 0.5, 0)

    def apply_canny(self,image, threshold1, threshold2):
        """
        This function is used to apply canny filter.
        :param image:
        :param threshold1: x-axis threshold
        :param threshold2: y-axis threshold
        :return: image with only edges
        """
        return cv2.Canny(image, threshold1, threshold2)

    def apply_connected_components(self,image, connectivity):
        """
        This function is used to apply connected components.
        :param image:
        :param connectivity: maximum number of components
        :return:
        """
        return cv2.connectedComponents(image, connectivity)

    def apply_dfs_fill_avg_color(self, binary_img, colored_img, strategy ='median'):
        """
        This function is used to apply dfs fill average color.
        :param binary_img: binary image
        :param colored_img: colored image
        :param strategy: mean or median
        :return: image with filled color
        """

        def dfs(node, visited, component):
            """
            This function is used to apply dfs.
            :param node: node
            :param visited: visited node
            :param component: component
            :return:
            """
            visited[x, y] = True
            component.append((x, y))

            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                if 0 <= x + dx < binary_img.shape[0] and 0 <= y + dy < binary_img.shape[1] and not visited[x, y] and binary_img[x, y] != 0:
                    dfs((x + dx, y + dy), visited, component)

        visited = np.zeros_like(binary_img, dtype=bool)
        output_img = np.zeros_like(colored_img)

        for x in range(binary_img.shape[0]):
            for y in range(binary_img.shape[1]):
                if not visited[x, y] and binary_img[x, y] == 0:
                    component = []
                    dfs((x, y), visited, component)
                    if component:
                        if strategy == 'mean':
                            avg_color = np.mean([colored_img[cx, cy] for cx, cy in component], axis=0)
                        else:
                            avg_color = np.median([colored_img[cx, cy] for cx, cy in component], axis=0)
                        for cx, cy in component:
                            output_img[cx, cy] = avg_color

        return output_img

    def quantize_image(self, image, n_colors=12):
        """
        This function is used to quantize image.
        :param image: image
        :param n_colors: number of colors
        :return: quantized image
        """
        # Convert image to data points
        data = np.float32(image).reshape((-1, 3))

        kmeans = KMeans(n_clusters=n_colors)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_

        quantized_data = np.float32([centers[label] for label in labels])

        quantized_image = quantized_data.reshape(image.shape)

        return np.uint8(quantized_image)

    def stauration_enhancement(self,image):
        """
        This function is used to enhance saturation.
        :param image: image
        :return: image with enhanced saturation
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv_image)

        s_equalized = cv2.equalizeHist(s)

        enhanced_hsv_image = cv2.merge([h, s_equalized, v])

        enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)

        return enhanced_image

    def colorize(self):
        """
        This function is used to colorize the image.
        :return: colorized image
        """

        smoothed_image = self.apply_gaussian_filter(3, 0)
        dog = self.difference_of_gaussian(3)
        edges_2 = self.apply_canny(smoothed_image,10, 40)
        edges_3 = cv2.bitwise_or(edges_2, dog)
        n, components = self.apply_connected_components(edges_3,8)

        L = np.zeros(smoothed_image.shape, np.uint8)
        mask_ = np.array(components, dtype=np.uint8)

        for label in range(1, n):
            if len(mask_[components == label]) > 20:
                mask_[components == label] = 255
                mask_[mask_ < 255] = 0
                L = cv2.bitwise_or(L, mask_)

        result = L

        resized_colored_img = cv2.resize(self.colorized_output, (result.shape[1], result.shape[0]))
        c_b, c_g, c_r = cv2.split(resized_colored_img)


        res_b = self.apply_dfs_fill_avg_color(result, c_b)
        res_g = self.apply_dfs_fill_avg_color(result, c_g)
        res_r = self.apply_dfs_fill_avg_color(result, c_r)
        final_img = cv2.merge([res_b, res_g, res_r])

        final_img = self.stauration_enhancement(final_img)
        final_img = self.quantize_image(final_img, 12)

        return final_img



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=Path, default="gen_model.pt")
    parser.add_argument("--test-image", type=Path, default="test\sketch.jpg")
    parser.add_argument("--test-output", type=Path, default='final_generated_image.jpg')
    args = parser.parse_args()
    MangaColorization(args.test_image,args.test_output,args.checkpoint_file)