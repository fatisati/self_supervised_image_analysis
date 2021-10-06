import os

data_path = '../data/ISIC/dermoscopic/'
image_path = 'resized255/'
mask_path = 'ISIC2018_Task2_Training_GroundTruth_v3/'
res = []
for file in os.listdir(data_path + mask_path):

    if 'ISIC_0000000' in file:
        part1 = file.split('attribute_')
        print(f'{part1}- part1')

        part2 = part1[-1].split('.')
        print(f'{part2}- part2')
        res.append(part2[0])
print(res)
print(','.join(res))