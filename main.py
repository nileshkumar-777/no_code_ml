# ==========================================================
#                    MAIN ENTRY POINT
# ==========================================================

from trainer import train
from inference import run_inference, list_available_models


# ==========================================================
#                         CLI MENU
# ==========================================================

if __name__ == "__main__":

    while True:

        print("\n=========== ML ENGINE ===========")
        print("1. Train Model")
        print("2. Run Inference")
        print("3. List Available Models")
        print("4. Exit")

        choice = input("Select option: ")

        # ==================================================
        # TRAIN MODEL
        # ==================================================

        if choice == "1":

            csv_path = input("Enter CSV path: ")
            target_col = input("Enter target column: ")

            train(csv_path, target_col)

        # ==================================================
        # RUN INFERENCE
        # ==================================================

        elif choice == "2":

            csv_path = input("Enter inference CSV path: ")
            run_inference(csv_path)

        # ==================================================
        # LIST MODELS
        # ==================================================

        elif choice == "3":

            list_available_models()

        # ==================================================
        # EXIT
        # ==================================================

        elif choice == "4":

            print("Exiting ML Engine.")
            break

        else:
            print("Invalid option.")