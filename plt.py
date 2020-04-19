import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
with open('/media/ding/Study/graduate/DABNet/checkpoint/paris/DABNetbs16gpu1_train/2020-03-29_22:37:40/log.txt', 'r') as f:
    next(f) 
    next(f) 
    lines = f.readlines()
    print(lines[100])
    print(lines[100].split('\t'))
    print(lines[100].split('\t')[0])
    print(lines[100].split('\t')[4])
    print(lines[100].split('\t')[7])
epoch = []
loss = []
miou = []
for i,line in enumerate(lines):
    # print(line.split('\t')[0].strip())
    # print(line.split('\t')[4].strip())
    epoch.append(line.split('\t')[0].strip())
    loss.append(line.split('\t')[4].strip())
    if i % 10 == 0 or i == 399:
        # print(line.split('\t')[7].strip())
        miou.append(line.split('\t')[7].strip())
print(len(loss))
fig1, ax1 = plt.subplots(figsize=(11, 8))    
ax1.plot(epoch, loss)
ax1.set_title("Average training loss vs epochs")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Current loss")
x_major_locator=MultipleLocator(50)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.05)
#把y轴的刻度间隔设置为10，并存在变量里
# plt.xlim(-0.5,11)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(0.1,0.7)


plt.savefig("loss_vs_epochs.png")

plt.clf()

ig2, ax2 = plt.subplots(figsize=(11, 8))

ax2.plot(epoch, miou, label="Val IoU")
ax2.set_title("Average IoU vs epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Current IoU")
plt.legend(loc='lower right')

plt.savefig("iou_vs_epochs.png")

plt.close('all')