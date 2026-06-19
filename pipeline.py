import torch
import os
from tqdm import tqdm
from utils import (
    confusion_matrix,
    validation_mtkd,
    validation_kd,
    validation,
    train_mtkd,
    train_kd,
    train
)
from models import AttentionMTKD
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def pipeline_ft(model, train_loader, valid_loader, test_loader, hyperparam, device):
    SESSION = hyperparam['SESSION']
    LEARNING_RATE = hyperparam['LEARNING_RATE']
    TRAINING = hyperparam['TRAINING']
    N_EPOCHS = hyperparam['N_EPOCHS']
    LINGUALITY = hyperparam['LINGUALITY']
    LANGUAGE = hyperparam['LANGUAGE']

    # file_path = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/ft/FT_{LINGUALITY}_{LANGUAGE}_S{SESSION}.pth"
    file_path = f"/content/FT_{LINGUALITY}_{LANGUAGE}_S{SESSION}.pth"

    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch = 1
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        total_train_loss = checkpoint['training loss']
        total_valid_loss = checkpoint['validation loss']
        print("Model checkpoint has been loaded")
    
    # Initialize TensorBoard writer
    # writer = SummaryWriter(log_dir=f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/tensorboards/FT_{LINGUALITY}_{LANGUAGE}_S{SESSION}")
    writer = SummaryWriter(log_dir=f"/content/tensorboards/FT_{LINGUALITY}_{LANGUAGE}_S{SESSION}")

    if TRAINING == 0: # No training, test/validation only
        test_unweighted_recall, test_weighted_recall, test_accuracy, total_test_loss, test_confidence_scores, all_actual_labels, all_predictions = validation(
            model,
            tqdm(test_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Test", leave=False),
            loss_fn, 
            device
        )
        print(f"Epoch {epoch}/{N_EPOCHS}, Test Recall (unweighted): {test_unweighted_recall}, Recall (weighted): {test_weighted_recall}, Accuracy: {test_accuracy}, Loss: {total_test_loss}\n")
        
        print(f"\n\nConfusion Matrix (S - {SESSION})\n")
        cm = confusion_matrix(all_actual_labels, all_predictions)
        print(cm)

    else: # Train first, then test/validation
        for epoch in range(epoch, N_EPOCHS+1):
            train_unweighted_recall, train_weighted_recall, train_accuracy, total_train_loss, train_confidence_scores = train(
                model,
                tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Training", leave=False),
                optimizer, 
                loss_fn, 
                device
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Training Recall (unweighted): {train_unweighted_recall}, Recall (weighted): {train_weighted_recall}, Accuracy: {train_accuracy}, Loss: {total_train_loss}")

            # Log training metrics to TensorBoard
            writer.add_scalar("Loss/Train", total_train_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Train", train_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Train", train_weighted_recall, epoch)


            valid_unweighted_recall, valid_weighted_recall, valid_accuracy, total_valid_loss, valid_confidence_scores, all_actual_labels, all_predictions = validation(
                model,
                tqdm(valid_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Validation", leave=False),
                loss_fn, 
                device
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Validation Recall (unweighted): {valid_unweighted_recall}, Validation Recall (weighted): {valid_weighted_recall}, Validation Accuracy: {valid_accuracy}, Validation Loss: {total_valid_loss}\n")

            # Log test metrics to TensorBoard
            writer.add_scalar("Loss/Validation", total_valid_loss, epoch)
            writer.add_scalar("Accuracy/Validation", valid_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Validation", valid_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Validation", valid_weighted_recall, epoch)

            writer.flush()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training loss': total_train_loss,
                'validation loss': total_valid_loss
            }, file_path)
            
            print(f"Model has been saved after epoch: {epoch}\n")

        # Close TensorBoard writer
        writer.close()

##########################################################################

def pipeline_kd(teacher, student, train_loader, valid_loader, test_loader, hyperparam, device):
    SESSION = hyperparam['SESSION']
    LEARNING_RATE = hyperparam['LEARNING_RATE']
    TRAINING = hyperparam['TRAINING']
    N_EPOCHS = hyperparam['N_EPOCHS']
    LINGUALITY = hyperparam['LINGUALITY']
    LANGUAGE = hyperparam['LANGUAGE']

    file_path_student = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/kd/KD_{LINGUALITY}_{LANGUAGE}_S{SESSION}.pth"

    optimizer_ce = torch.optim.AdamW(student.parameters(), lr = LEARNING_RATE)
    optimizer_kl = torch.optim.AdamW(student.parameters(), lr = LEARNING_RATE)
    loss_fn_ce = torch.nn.CrossEntropyLoss()
    loss_fn_kl = torch.nn.KLDivLoss(reduction='mean')

    epoch = 1
    if os.path.exists(file_path_student):
        checkpoint = torch.load(file_path_student)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ce.load_state_dict(checkpoint['optimizer_ce_state_dict'])
        optimizer_kl.load_state_dict(checkpoint['optimizer_kl_state_dict'])
        epoch = checkpoint['epoch']
        total_train_loss = checkpoint['training loss']
        total_valid_loss = checkpoint['validation loss']
        print("Student model checkpoint has been loaded")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/tensorboards/KD_{LINGUALITY}_{LANGUAGE}_S{SESSION}")

    lambda_param=0.75

    if TRAINING == 0: # No training, test only
        test_unweighted_recall, test_weighted_recall, test_accuracy, total_test_loss, total_kl_loss, total_ce_loss, test_confidence_scores, all_actual_labels, all_predictions = validation_kd(
            teacher, student,
            tqdm(test_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Validation", leave=False),
            loss_fn_ce, loss_fn_kl,
            device,
            lambda_param=lambda_param
        )
        print(f"Epoch {epoch}/{N_EPOCHS}, Test Recall (unweighted): {test_unweighted_recall}, Recall (weighted): {test_weighted_recall}, Accuracy: {test_accuracy}, Loss: {total_test_loss}, KLD Loss: {total_kl_loss}, CE Loss: {total_ce_loss}\n")
    
        print(f"\n\nConfusion Matrix (S - {SESSION})\n")
        cm = confusion_matrix(all_actual_labels, all_predictions)
        print(cm)

    else: # Train first, then validation
        best_test_unweighted_recall = 0
        for epoch in range(epoch, N_EPOCHS+1):
            train_unweighted_recall, train_weighted_recall, train_accuracy, total_train_loss, total_kl_loss, total_ce_loss, train_confidence_scores = train_kd(
                teacher, student,
                tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Training", leave=False),
                optimizer_ce, optimizer_kl, 
                loss_fn_ce, loss_fn_kl,
                device,
                lambda_param=lambda_param
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Training Recall (unweighted): {train_unweighted_recall}, Recall (weighted): {train_weighted_recall}, Accuracy: {train_accuracy}, Loss: {total_train_loss}, KLD Loss: {total_kl_loss}, CE Loss: {total_ce_loss}")

            # Log training metrics to TensorBoard
            writer.add_scalar("Loss/Train", total_train_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Train", train_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Train", train_weighted_recall, epoch)

            test_unweighted_recall, test_weighted_recall, test_accuracy, total_test_loss, total_kl_loss, total_ce_loss, test_confidence_scores, all_actual_labels, all_predictions = validation_kd(
                teacher, student,
                tqdm(valid_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Test", leave=False),
                loss_fn_ce, loss_fn_kl,
                device,
                lambda_param=lambda_param
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Validation Recall (unweighted): {test_unweighted_recall}, Recall (weighted): {test_weighted_recall}, Accuracy: {test_accuracy}, Loss: {total_test_loss}, KLD Loss: {total_kl_loss}, CE Loss: {total_ce_loss}\n")
            
            # Log test metrics to TensorBoard
            writer.add_scalar("Loss/Validation", total_test_loss, epoch)
            writer.add_scalar("Accuracy/Validation", test_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Validation", test_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Validation", test_weighted_recall, epoch)

            writer.flush()

            if test_unweighted_recall > best_test_unweighted_recall:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_ce_state_dict': optimizer_ce.state_dict(),
                    'optimizer_kl_state_dict': optimizer_kl.state_dict(),
                    'training loss': total_train_loss,
                    'validation loss': total_test_loss
                }, file_path_student)
                
                print(f"Model has been saved after epoch: {epoch}\n")
                best_test_unweighted_recall = test_unweighted_recall

    # Close TensorBoard writer
    writer.close()

##########################################################################

def pipeline_mtkd(teacher_en, teacher_fi, teacher_fr, student, train_loader, valid_loader, test_loader, hyperparam, device):
    SESSION = hyperparam['SESSION']
    LEARNING_RATE = hyperparam['LEARNING_RATE']
    TRAINING = hyperparam['TRAINING']
    N_EPOCHS = hyperparam['N_EPOCHS']
    LINGUALITY = hyperparam['LINGUALITY']
    LANGUAGE = hyperparam['LANGUAGE']

    # Read configuration parameters
    teacher_selection = hyperparam.get('teacher_selection', 'cosine')
    ce_weight = hyperparam.get('ce_weight', 1.0)
    kd_weight = hyperparam.get('kd_weight', 1.0)
    contrastive_weight = hyperparam.get('contrastive_weight', 0.0)
    contrastive_temp = hyperparam.get('contrastive_temp', 0.07)
    attention_hidden_dim = hyperparam.get('attention_hidden_dim', 16)

    file_path_student = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/mtkd/MTKD_{LINGUALITY}_{LANGUAGE}_S{SESSION}.pth"

    # Initialize attention network if active
    attention_net = None
    if teacher_selection == "attention":
        attention_net = AttentionMTKD(
            num_classes=student.config.num_labels, 
            hidden_dim=attention_hidden_dim, 
            num_teachers=3
        )
        attention_net.to(device)

    # Set up optimizer parameters to include attention network if training with attention
    if attention_net is not None:
        params = list(student.parameters()) + list(attention_net.parameters())
    else:
        params = list(student.parameters())

    optimizer_ce = torch.optim.AdamW(params, lr = LEARNING_RATE)
    optimizer_kl = torch.optim.AdamW(params, lr = LEARNING_RATE)
    loss_fn_ce = torch.nn.CrossEntropyLoss()
    loss_fn_kl = torch.nn.KLDivLoss(reduction='mean')

    epoch = 1
    if os.path.exists(file_path_student):
        checkpoint = torch.load(file_path_student)
        student.load_state_dict(checkpoint['model_state_dict'])
        if attention_net is not None and 'attention_net_state_dict' in checkpoint:
            attention_net.load_state_dict(checkpoint['attention_net_state_dict'])
        optimizer_ce.load_state_dict(checkpoint['optimizer_ce_state_dict'])
        optimizer_kl.load_state_dict(checkpoint['optimizer_kl_state_dict'])
        epoch = checkpoint['epoch']
        total_train_loss = checkpoint['training loss']
        total_valid_loss = checkpoint['validation loss']
        print("Student model checkpoint has been loaded")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/tensorboards/MTKD_{LINGUALITY}_{LANGUAGE}_S{SESSION}")

    if TRAINING == 0: # No training, test/validation only
        (
            test_unweighted_recall, test_weighted_recall, test_accuracy, 
            total_test_loss, total_kl_loss, total_ce_loss, total_contrastive_loss, 
            test_confidence_scores, all_actual_labels, all_predictions,
            val_attention_weights, epoch_attention_weights
        ) = validation_mtkd(
            teacher_en, teacher_fi, teacher_fr, student,
            tqdm(test_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Test", leave=False),
            loss_fn_ce, loss_fn_kl,
            device,
            attention_net=attention_net,
            teacher_selection=teacher_selection,
            ce_weight=ce_weight,
            kd_weight=kd_weight,
            contrastive_weight=contrastive_weight,
            contrastive_temp=contrastive_temp
        )
        print(f"Epoch {epoch}/{N_EPOCHS}, Test Recall (unweighted): {test_unweighted_recall}, Test Recall (weighted): {test_weighted_recall}, Test Accuracy: {test_accuracy}, Test Loss: {total_test_loss}, KLD Loss: {total_kl_loss}, CE Loss: {total_ce_loss}, SCL Loss: {total_contrastive_loss}\n")
        print(f"Average Attention Weights during Validation: EN = {val_attention_weights[0]:.4f}, FI = {val_attention_weights[1]:.4f}, FR = {val_attention_weights[2]:.4f}")
        
        print(f"\n\nConfusion Matrix (S - {SESSION})\n")
        cm = confusion_matrix(all_actual_labels, all_predictions)
        print(cm)
    
    else: # Train first, then test/validation
        best_test_unweighted_recall = 0
        for epoch in range(epoch, N_EPOCHS+1):
            (
                train_unweighted_recall, train_weighted_recall, train_accuracy, 
                total_train_loss, total_kl_loss, total_ce_loss, total_contrastive_loss, 
                train_confidence_scores, train_attention_weights
            ) = train_mtkd(
                teacher_en, teacher_fi, teacher_fr, student,
                tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Training", leave=False),
                optimizer_ce, optimizer_kl, 
                loss_fn_ce, loss_fn_kl,
                device,
                attention_net=attention_net,
                teacher_selection=teacher_selection,
                ce_weight=ce_weight,
                kd_weight=kd_weight,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Training Recall (unweighted): {train_unweighted_recall:.4f}, Recall (weighted): {train_weighted_recall:.4f}, Accuracy: {train_accuracy:.4f}, Loss: {total_train_loss:.4f}, KLD Loss: {total_kl_loss:.4f}, CE Loss: {total_ce_loss:.4f}, SCL Loss: {total_contrastive_loss:.4f}")
            print(f"Average Attention Weights: EN = {train_attention_weights[0]:.4f}, FI = {train_attention_weights[1]:.4f}, FR = {train_attention_weights[2]:.4f}")

            # Log training metrics to TensorBoard
            writer.add_scalar("Loss/Train", total_train_loss, epoch)
            writer.add_scalar("Loss/Train/CE", total_ce_loss, epoch)
            writer.add_scalar("Loss/Train/KL", total_kl_loss, epoch)
            writer.add_scalar("Loss/Train/Contrastive", total_contrastive_loss, epoch)
            writer.add_scalar("Weights/Train/EN", train_attention_weights[0], epoch)
            writer.add_scalar("Weights/Train/FI", train_attention_weights[1], epoch)
            writer.add_scalar("Weights/Train/FR", train_attention_weights[2], epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Train", train_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Train", train_weighted_recall, epoch)

            (
                test_unweighted_recall, test_weighted_recall, test_accuracy, 
                total_test_loss, total_kl_loss_val, total_ce_loss_val, total_contrastive_loss_val, 
                test_confidence_scores, all_actual_labels, all_predictions,
                val_attention_weights, epoch_attention_weights
            ) = validation_mtkd(
                teacher_en, teacher_fi, teacher_fr, student,
                tqdm(valid_loader, desc=f"Epoch {epoch}/{N_EPOCHS}, Validation", leave=False),
                loss_fn_ce, loss_fn_kl,
                device,
                attention_net=attention_net,
                teacher_selection=teacher_selection,
                ce_weight=ce_weight,
                kd_weight=kd_weight,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp
            )
            print(f"Epoch {epoch}/{N_EPOCHS}, Validation Recall (unweighted): {test_unweighted_recall:.4f}, Recall (weighted): {test_weighted_recall:.4f}, Accuracy: {test_accuracy:.4f}, Loss: {total_test_loss:.4f}, KLD Loss: {total_kl_loss_val:.4f}, CE Loss: {total_ce_loss_val:.4f}, SCL Loss: {total_contrastive_loss_val:.4f}\n")
            print(f"Validation Average Attention Weights: EN = {val_attention_weights[0]:.4f}, FI = {val_attention_weights[1]:.4f}, FR = {val_attention_weights[2]:.4f}")

            # Log test/validation metrics to TensorBoard
            writer.add_scalar("Loss/Validation", total_test_loss, epoch)
            writer.add_scalar("Loss/Validation/CE", total_ce_loss_val, epoch)
            writer.add_scalar("Loss/Validation/KL", total_kl_loss_val, epoch)
            writer.add_scalar("Loss/Validation/Contrastive", total_contrastive_loss_val, epoch)
            writer.add_scalar("Weights/Validation/EN", val_attention_weights[0], epoch)
            writer.add_scalar("Weights/Validation/FI", val_attention_weights[1], epoch)
            writer.add_scalar("Weights/Validation/FR", val_attention_weights[2], epoch)
            writer.add_scalar("Accuracy/Validation", test_accuracy, epoch)
            writer.add_scalar("Recall_Unweighted/Validation", test_unweighted_recall, epoch)
            writer.add_scalar("Recall_Weighted/Validation", test_weighted_recall, epoch)

            writer.flush()

            if test_unweighted_recall > best_test_unweighted_recall:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_ce_state_dict': optimizer_ce.state_dict(),
                    'optimizer_kl_state_dict': optimizer_kl.state_dict(),
                    'training loss': total_train_loss,
                    'validation loss': total_test_loss
                }
                if attention_net is not None:
                    save_dict['attention_net_state_dict'] = attention_net.state_dict()
                
                # Save student checkpoint
                torch.save(save_dict, file_path_student)
                print(f"Model checkpoint has been saved after epoch: {epoch}\n")
                best_test_unweighted_recall = test_unweighted_recall

                # Save validation attention weights for visualization
                vis_dir = os.path.join(os.path.dirname(file_path_student), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                weights_save_path = os.path.join(
                    vis_dir, 
                    f"attention_weights_{LINGUALITY}_{LANGUAGE}_S{SESSION}_epoch{epoch}.pt"
                )
                torch.save(epoch_attention_weights, weights_save_path)
                print(f"Saved validation attention weights for visualization to {weights_save_path}")

    # Close TensorBoard writer
    writer.close()


##########################################################################
    
