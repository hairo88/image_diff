import matplotlib.pyplot as plt

def show_2Img(image1, image2, title):
    # 画像を表示
    plt.figure(figsize=(10, 5))

    # 画像1
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(f'{title} 1')
    plt.axis('off')

    # 画像2
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(f'{title} 2')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

