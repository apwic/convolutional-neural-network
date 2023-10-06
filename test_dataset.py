from utils.ImageToMatrix import getDataset

def main():
    # TODO: gimana caranya ini masuk cuk ke sekwensial kt
    train_data, val_data, test_data = getDataset()
    
    print(train_data)
    
if __name__ == '__main__':
    main()