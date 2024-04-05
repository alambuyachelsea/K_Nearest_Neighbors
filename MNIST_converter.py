# Converts the data sets into CSV files and saves them

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    label = open(labelf, "rb")

    f.read(16)
    label.read(8)
    images = []

    for i in range(n):
        image = [ord(label.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    label.close()


convert("mnist_samples/train-images-idx3-ubyte",
        "mnist_samples/train-labels-idx1-ubyte",
        "mnist_train.csv", 10000)
convert("mnist_samples/t10k-images-idx3-ubyte",
        "mnist_samples/t10k-labels-idx1-ubyte",
        "mnist_test.csv", 1000)
