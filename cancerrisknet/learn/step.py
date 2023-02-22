import torch
import torch.nn.functional as F


def get_model_loss(logit, batch, args):
    y_seq = batch['y_seq']
    y_mask = batch['y_mask']
    if args.loss_fn == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), reduction='sum')\
               / torch.sum(y_mask.float())
    elif args.loss_fn == 'mse':
        loss = F.mse_loss(logit, y_seq.float(), size_average=False, reduction='mean')
    else:
        raise Exception('Loss function is illegal or not found.')
    return loss


def model_step(batch, models, train_model, args):
    """
        Single step of running model on the a batch x,y and computing the loss.
        Returns various stats of this single forward and backward pass.

        Args:
            batch: whole batch dict, can be used by various special args
            models: dict of models. The main model, named "model" must return logit, hidden, activ
            train_model: Backward pass is computed if set to True.

        Returns:
            loss: scalar for loss on batch as a tensor
            preds: predicted labels as numpy array
            probs: softmax probablities as numpy array
            golds: labels at the trajectory level, numpy array version of arg y
            patient_golds: labels at the patient level
            exams: exam ids for batch if available as a list of strings
            pids: deidentified patient ids as a list of strings
            censor_times: feature rep for batch
            days_to_censor: the time before censorship as a tensor
            dates: the admission data as a tensor
    """
    logit = models[args.model_name](batch['x'], batch)
    loss = get_model_loss(logit, batch, args)

    if train_model:
        loss.backward()

    probs = torch.sigmoid(logit).cpu().data.numpy()  # Shape is B, len(args.month_endpoints)
    preds = probs > .5
    golds = batch['y'].data.cpu().numpy()
    patient_golds = batch['future_panc_cancer'].data.cpu().numpy()
    exams = batch['exam']
    pids = batch['patient_id']
    censor_times = batch['time_at_event'].cpu().numpy()
    days_to_censor = batch['days_to_censor'].cpu().numpy()
    dates = batch['admit_date']

    return loss, preds, probs, golds, patient_golds, exams, pids, censor_times, days_to_censor, dates
