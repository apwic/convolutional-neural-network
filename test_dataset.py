from utils.ImageToMatrix import getDataset

def main():
    train_data, val_data, test_data = getDataset()
    
    print(train_data)
    
if __name__ == '__main__':
    main()