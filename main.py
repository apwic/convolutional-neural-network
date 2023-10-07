import argparse
import helper

def main():
    parser = argparse.ArgumentParser(description="Neural Network Training and Loading", formatter_class=argparse.RawTextHelpFormatter)

    # Subparsers for "run" command
    subparsers = parser.add_subparsers(dest="command", title="Commands", description="Valid commands to run the program")

    # Parser for the "run" command
    run_parser = subparsers.add_parser("run", help="Run the training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Group for training arguments
    training_group = run_parser.add_argument_group("Training Arguments")
    training_group.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs")
    training_group.add_argument("-b", "--batch_size", type=int, default=5, help="Batch size")
    training_group.add_argument("-n", "--num_samples", type=int, default=10, help="Length of split dataset")
    training_group.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Length of split dataset")
    
    # Group for model management
    model_group = run_parser.add_argument_group("Model Management")
    model_group.add_argument("-s", "--save", type=str, help="Save the model after training")
    model_group.add_argument("-l", "--load", type=str, help="Name of the file to load (including .json)")

    args = parser.parse_args()

    if args.command == "run":
        if hasattr(args, 'l') and args.l:
            model = helper.createModel(load_file=args.l)
        else:
            model = helper.createModel()

        kwargs = {
            'num': args.num_samples,
            'epoch': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }

        helper.run(model=model, **kwargs)
        
        if hasattr(args, 's') and args.s:
            model.saveModel(filename=args.s)

if __name__ == "__main__":
    main()