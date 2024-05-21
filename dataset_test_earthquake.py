from earthquake_data_loader import EarthquakeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

train_data = EarthquakeDataset("train")
test_data = EarthquakeDataset("test")

train_loader = DataLoader(train_data,batch_size=1, shuffle=False)
test_loader = DataLoader(test_data,batch_size=1, shuffle=False)

print("\nTraining Data")
for batch_idx, data in enumerate(train_loader):
    print(f"\nData loaded ({batch_idx}), time current ({data[3]}), Significance ({data[4]}), Magnitudo ({data[5]})")
    for i in range(len(data[0])):
        if(i==0):
            print("\tEvent node : %d , delta_days: %s, significance: %f, magnitudo %f" %
                   (int(data[0][i].item()),f"[Day: {data[1][0][i][0]}, Hours: {data[1][0][i][1]}, Minutes: {data[1][0][i][2]}, Seconds: {data[1][0][i][3]}]",float(data[4]),float(data[5])))
        else:
            print("\tNeigh node : %d , delta_days: %s, significance: %f, magnitudo %f" %
                   (int(data[0][i].item()),f"[Day: {data[1][0][i][0]}, Hours: {data[1][0][i][1]}, Minutes: {data[1][0][i][2]}, Seconds: {data[1][0][i][3]}]",float(data[4]),float(data[5])))
    if batch_idx == 9:
        break
print("\nTest Data")
for batch_idx, data in enumerate(test_loader):
    print(f"\nData loaded ({batch_idx}), time current ({data[3]}), Significance ({data[4]}), Magnitudo ({data[5]})")
    for i in range(len(data[0])):
        if(i==0):
            print("\tEvent node : %d , delta_days: %s, significance: %f, magnitudo %f" %
                   (int(data[0][i].item()),f"[Day: {data[1][0][i][0]}, Hours: {data[1][0][i][1]}, Minutes: {data[1][0][i][2]}, Seconds: {data[1][0][i][3]}]",float(data[4]),float(data[5])))
        else:
            print("\tNeigh node : %d , delta_days: %s, significance: %f, magnitudo %f" %
                   (int(data[0][i].item()),f"[Day: {data[1][0][i][0]}, Hours: {data[1][0][i][1]}, Minutes: {data[1][0][i][2]}, Seconds: {data[1][0][i][3]}]",float(data[4]),float(data[5])))
    if batch_idx == 9 :
        break
# return u, time_delta, time_bar, time_cur,significance,magnitudo
