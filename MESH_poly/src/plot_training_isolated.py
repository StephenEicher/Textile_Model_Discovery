
def plot_training(dfs, Psi_model_type, modelFit_mode, p, path2saveResults, best_reg=2, is_I4beta=True):
    """
    Create various figures that show the final model fit and how it changed throughout the training process
    :param dfs: dataframe used for training
    :param Psi_model_type: Function that is called to instantiate model
    :param modelFitMode: String corresponding to which loading directions are used for training
    :param p: p value used for lp regularization
    :param path2saveResults: Path to save created figures and animations to
    :param best_reg: index that specifies which regularization penalty to use for the best fit graph (i.e. the best fit graph will be plotted using alpha = Lalphas[best_reg])
    """

    # Load data from file
    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    Lalphas = input_data["Lalphas"]
    Region = input_data["Region"]
    P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss, midpoint = getStressStrain(dfs, Region)

    Psi_model, terms = Psi_model_type(lam_ut_all, gamma_ss, P_ut, P_ss, modelFit_mode, 0, True, p)
    is_noiso = (terms < 14)
    model_UT, model_SS, Psi_model, model = modelArchitecture(Region, Psi_model)


    # Flatten first 2 dims of weight_hist_arr so first dimension is just total # of epochs
    weight_hist_arr = [x for y in full_weight_hist for x in y]
    n_epochs_per_lalpha = [len(x) for x in full_weight_hist] # Total number of epochs per regularization weight
    # Get total # of epochs elapset at start of each regularization weight
    first_epoch_per_lalpha = [sum(n_epochs_per_lalpha[0:i]) for i in range(len(n_epochs_per_lalpha))]

    # Plot Best Fit
    plt.rcParams['figure.figsize'] = [35, 12]
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = "\n".join([
    #     r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
    #     r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
    #     r'\usepackage[default]{sourcesanspro}',  # set the normal font here
    #     r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
    #     r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    # ])
    fig, axes = plt.subplots(1, 2)
    (ax_w, ax_s) = axes
    inputs = reshape_input_output_mesh(lam_ut)
    outputs = reshape_input_output_mesh(P_ut)
    Psi_model.set_weights(weight_hist_arr[first_epoch_per_lalpha[best_reg + 1] if len(first_epoch_per_lalpha) > best_reg + 1 else len(weight_hist_arr) - 1])
    preds = reshape_input_output_mesh(model_UT.predict(lam_ut_all))
    cmap = plt.cm.get_cmap('jet_r', 5)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # labels_ws = ["X", "Y", "Biax", "N/A", "N/A"]
    labels_ws = ["X", "Y", "Biax"]
    # labels_45 = ["N/A", "N/A", "N/A", "N/A", "N/A"]

    num_points = 17
    for i in range(3):
        ax_w.scatter(resample(inputs[0][i][0], num_points), resample(outputs[0][i][0], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        ax_s.scatter(resample(inputs[0][i][1], num_points), resample(outputs[0][i][1], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        # ax_45.scatter(resample(inputs[1][i][0], num_points), resample(outputs[1][i][0], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        ax_w.plot(inputs[0][i][0], preds[0][i][0], color=cmaplist[i], label=labels_ws[i], zorder=25, lw=6)
        ax_s.plot(inputs[0][i][1], preds[0][i][1], color=cmaplist[i], label=labels_ws[i], zorder=25, lw=6,)
        # ax_45.plot(inputs[1][i][0], preds[1][i][0], color=cmaplist[i], label=labels_45[i], zorder=25, lw=6,)

    ax_w.set_xlabel("X stretch [-]")
    ax_w.set_ylabel("X stress [kPa]")
    ax_s.set_xlabel("Y stretch[-]")
    ax_s.set_ylabel("Y stress [kPa]")
    # ax_45.set_xlabel("x stretch [-]")
    # ax_45.set_ylabel("x stress [kPa]")

    ax_w.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)
    ax_s.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)
    # ax_45.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)

    plt.tight_layout(pad=1)
    plt.savefig(f"{path2saveResults}/best_fit.pdf",
                transparent=False,
                facecolor='white')

    # Create gif of model training across epochs
    # Create figure with multiple subplots to show state of model at a given epoch
    plt.rcParams['figure.figsize'] = [30, 18 if is_noiso else 20]
    plt.rcParams['figure.constrained_layout.use'] = True

    if Region.startswith('mesh'):
        fig, axes = plt.subplots(3, 5)
        axes = flatten([list(x) for x in zip(*axes[0:2])]) + list(axes[2])
        direction_strings = ["X", "Y"] * 5 + ["x", "y", "x", "y", "x"]
        n_spaces = 55
        titles = ["X", None, "Y", None, "Biax", None, "N/A", None, "N/A", None, " " * n_spaces + "N/A", None," " * n_spaces + "N/A", None, "N/A"]
        inputs = reshape_input_output_mesh(lam_ut)
        inputs = [[x[i] if np.max(x[i]) > 1.0 else ((x[1 - i] - 1) * 1e-9 + 1) for i in range(2)] for y in inputs for x in y] # 10 x 2
        inputs = flatten(inputs)[0:15] # Length 15 list
        outputs = flatten(flatten(reshape_input_output_mesh(P_ut)))[0:15]

        all_plots = [
            plotMap(axes[i], Psi_model, weight_hist_arr[0], DummyModel(), terms, inputs[i], outputs[i], direction_strings[i], titles[i])
            for i in range(len(axes))]  # Create dummy plots, will update in loop

        h, l = axes[0].get_legend_handles_labels()


        cmap = plt.cm.get_cmap('jet_r', 14)  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        last_inv = "I_{4s_{I, II}}" if is_I4beta else "I_{4s}"
        labels = [x for In in range(1, 3) for x in [f"$(I_{In} - 3)$", f"exp$( (I_{In} - 3))$", f"$(I_{In} - 3)^2$", f"exp$( (I_{In} - 3)^2)$"]]
        labels = labels + [x for dir in ["I_{4w}", last_inv] for x in [f"exp$({dir}) -  {dir}$", f"$({dir} - 1)^2$", f"exp$( ({dir} - 1)^2)$", ]]
        if len(labels) > terms: # Handle anisotropic only case
            labels = labels[(len(labels) - terms):]
            cmaplist = cmaplist[(len(cmaplist) - terms):]

        legend_handles = [Patch(color=c) for c in cmaplist] + [all_plots[0][-1]]
        labels += ["data"]
        leg = fig.legend(loc="lower center", ncols=4, handles=legend_handles, labels=labels,
                         mode="expand", fontsize=40)
        leg.get_frame().set_alpha(0)


    else:
        fig, axes = plt.subplots(1, 3)
        models = [model_SS, model_UT, model_UT]
        inputs = [gamma_ss, lam_ut_all[:(midpoint + 1)], lam_ut_all[midpoint:]]
        outputs = [P_ss, P_ut[:(midpoint + 1)], P_ut[midpoint:]]
        maxima = [np.max(output) for output in outputs]  # used for scaling plots
        all_plots = [
            plotMap(axes[i], Psi_model, weight_hist_arr[0], models[i], terms, inputs[i], outputs[i], "", "") for i
            in range(len(models))]

    # Size of both of these arrays is M x N x P where M is the number of time steps we are plotting at, N is the number
    # of terms in the model, and P is the number of plots
    all_paths = [] # List of all paths that define the shaded regions in the plots
    all_uppers = [] # List of all curves that define the boundaries between shaded regions

    # # Adjust axes for psi plots
    # for i in range(3, 6):
    #     axes[i].set_ylim(0, maxima[i])
    # fig.suptitle(f'$\\alpha$ = {Lalphas[0]} , epoch = {0}',
    #              fontsize=30, weight="bold")
    engine = fig.get_layout_engine()
    leg_height = 0.13 if is_noiso else 0.20
    engine.set(rect=(0.005, leg_height , 0.99, 0.995 - leg_height), wspace=0.04)
    # fig.tight_layout(pad=50)

    # plt.pause(1)  # renders all plots
    n_epochs = len(weight_hist_arr)
    n_frames = 0  # Typical size of gif
    if n_frames > 0:
        steps = [int(i * (n_epochs - 1) / n_frames) for i in range(n_frames + 1)] # list of epochs when we will plot
    else:
        steps = [x - 1 for x in first_epoch_per_lalpha]  + [n_epochs - 1]
        steps[0] = 0
    # Iterate through epochs and precompute the appropriate paths to draw


    for i in steps:
        model_weights = weight_hist_arr[i]
        lalpha_idx = sum([i >= x for x in first_epoch_per_lalpha]) - 1
        # Create length 6 list of M x N numpy arrays where M is the number of points in the
        predictions = [np.zeros([output.shape[0], terms]) for output in outputs]
        curr_path = []
        curr_upper = []
        # Compute contribution from one term at a time
        for term in range(terms):
            model_plot = GetZeroList(model_weights)
            if len(model_weights) > terms: # If number of weights is > number of terms, then it is alternate CANN
                # Make 2 weights (gain and exponent) nonzero at a time
                model_plot[2 * term] = model_weights[2 * term]
                model_plot[2 * term + 1] = model_weights[2 * term + 1]
            else:
                # Otherwise, make one weight nonzero at a time
                model_plot[term] = model_weights[term]

            Psi_model.set_weights(model_plot)

            # Add up all the terms BEFORE the current term to get the lower bound for the shaded region
            lowers = [np.sum(prediction, axis=1) for prediction in predictions]
            # Compute contribution from current term
            pred_shear = model_SS.predict(gamma_ss) if gamma_ss != [] else []
            pred_ut = model_UT.predict(lam_ut, verbose=0)
            for plot_id in range(len(predictions)):
                if Region.startswith('mesh'):
                    predictions[plot_id][:, term] = flatten(flatten(reshape_input_output_mesh(pred_ut)))[plot_id].flatten()
                else:
                    predictions[plot_id][:, term] = models[plot_id].predict(inputs[plot_id])[:].flatten()

            # Add up all the terms INCLUDING the current term to get the upper bound for the shaded region
            uppers = [np.sum(prediction, axis=1) for prediction in predictions]

            # Create a path from the upper and lower bounds and add to array
            paths = [create_verts(inputs[k], lowers[k].flatten(), uppers[k].flatten()) for k in range(len(inputs))]
            curr_path.append(paths)
            curr_upper.append(uppers)

        # Add array that has all the terms at the current timestep to another array
        all_paths.append(curr_path)
        all_uppers.append(curr_upper)

    # Set up limits properly
    for k in range(len(axes)):
        min_P = np.min(outputs[k])
        max_P = np.max(outputs[k])
        min_x = np.min(inputs[k])
        max_x = np.max(inputs[k])
        if np.max(inputs[k]) - np.min(inputs[k]) < 1e-6:
            axes[k].set_xticks([np.min(inputs[k]), np.max(inputs[k])])
            axes[k].set_xticklabels(['1', '1'])
        if abs(min_P) < abs(max_P):
            # Tension / Shear
            axes[k].set_xlim([min_x, max_x])
            axes[k].set_ylim([0.0, max_P])
        elif abs(min_P) > abs(max_P):
            # Compression
            axes[k].set_xlim([max_x, min_x])
            axes[k].set_ylim([0.0, min_P])
        else:
            axes[k].set_xlim([1, 2])
            axes[k].set_ylim([0.0, 1])
            axes[k].set_xticks([1, 2])
            axes[k].set_yticks([0, 1])


    # Once paths have been precomputed, begin animation
    for i in range(len(all_uppers)):
        # At each time step update all the plots
        for term in range(terms):
            for plot_id in range(len(all_plots)):
                all_plots[plot_id][2 * term].set_paths(all_paths[i][term][plot_id])
                all_plots[plot_id][2 * term + 1][0].set_ydata(all_uppers[i][term][plot_id])

        # Get current epoch number
        epoch_number = steps[i]
        lalpha_idx = sum([epoch_number >= x for x in first_epoch_per_lalpha]) - 1
        epoch_number = epoch_number - first_epoch_per_lalpha[lalpha_idx]
        # Put epoch # and whether there is regularization in compression plot title (so it is in the top middle)
        # fig.suptitle(f'$\\alpha$ = {Lalphas[lalpha_idx]} , epoch = {epoch_number}',
        #           fontsize=30, weight="bold")

        # for j in range(5):

        # Do not plot R2 == comment out if you want to
        for plot_id in range(len(axes)):
            if (steps[i] + 1) in first_epoch_per_lalpha:
                curr_reg = first_epoch_per_lalpha.index(steps[i] + 1) - 1
                r2 = input_data["r2"][0][curr_reg].flatten()[plot_id]
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel(f"$R^2$ = {r2:.4f}", labelpad=-50)
            elif steps[i] == n_epochs - 1:
                r2 = input_data["r2"][0][-1].flatten()[plot_id]
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel(f"$R^2$ = {r2:.4f}",
                                                                                            labelpad=-50)
            else:
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel("", labelpad=-50)
        ###########
        rect_height = 0.297 if is_noiso else 0.275
        # for j in range(3):
        #     rec = plt.Rectangle((0.2 * j, leg_height + rect_height), 0.2, 1 - leg_height - rect_height, fill=False, lw=2)
        #     rec.set_zorder(1000)
        #     rec = fig.add_artist(rec)
        # for j in range(2):
        #     rec = plt.Rectangle((0.4 * j, leg_height), 0.4, rect_height, fill=False, lw=2)
        #     rec.set_zorder(1000)
        #     rec = fig.add_artist(rec)
        # rec = plt.Rectangle((0.8, leg_height), 0.2, rect_height, fill=False, lw=2)
        # rec.set_zorder(1000)
        # rec = fig.add_artist(rec)
        # for term in range(terms):
        #     leg.get_texts()[term].set_text("%.2f" % weight_hist_arr[steps[i]][2 * term + 1])
        # fig.tight_layout(pad=0)
        # plt.subplots_adjust(top=0.95, bottom=0.20)
        #########
        # Render and save plot
        plt.savefig(f"{path2saveResults}/img_{i}.pdf",
                    transparent=False,
                    facecolor='white')
        plt.savefig(f"{path2saveResults}/img_{i}.png",
                    transparent=False,
                    facecolor='white')

    plt.close()

    # Create image from snapshots in training process
    first_step_per_lalpha = [[step >= first_epoch for step in steps].index(True) - 1 for first_epoch in first_epoch_per_lalpha]
    first_step_per_lalpha[0] = 0 # First image should be at epoch 0
    first_step_per_lalpha.append(len(steps) - 1)  # Last epoch should be final result
    extra_pad = 5
    training_img = None
    for i in range(len(first_step_per_lalpha)):
        # Load image and add to frame list
        path = f"{path2saveResults}/img_{first_step_per_lalpha[i]}.pdf"
        new_path = f"{path2saveResults}/training_{i}.pdf"
        shutil.copyfile(path, new_path)

        im = convert_from_path(path)[0]
        if i == 0:
            training_img = Image.new('RGB', (im.width * 2 + extra_pad, im.height * 4 + 3 * extra_pad))
        training_img.paste(im, ((i % 2) * (im.width + extra_pad), int(i / 2) * (im.height + extra_pad)))

    training_img.save(f"{path2saveResults}/training.pdf")

    # Repeat but formatted for paper
    if len(first_epoch_per_lalpha) > 1:
        plot_indices = [0, 5, 4, 9]
        num_regs = 2
        offset = 300
        for i in range(num_regs):
            # Load image and add to frame list
            path = f"{path2saveResults}/img_{first_step_per_lalpha[i + 1]}.pdf"
            im = convert_from_path(path, poppler_path='/opt/homebrew/Cellar/poppler/24.04.0/bin')[0]
            if i == 0:
                W = im.width
                H = im.height
                w = int(W / 5)
                h = int(H * 0.95 / 3)
                training_img = Image.new('RGB', (w * num_regs + (num_regs - 1) * extra_pad, h * len(plot_indices) + (len(plot_indices) - 1) * extra_pad))
            for j in range(len(plot_indices)):
                row = int(plot_indices[j] / 5)
                col = plot_indices[j] % 5
                im_crop = im.crop((col * w, H - h * (3 - row) - offset, (col + 1) * w, H - h * (2 - row) - offset))
                training_img.paste(im_crop, (i * (w + extra_pad), j * (h + extra_pad)))

        training_img.save(f"{path2saveResults}/regularization.pdf")

    # Create array of all frames in GIF
    frames = []
    for i in range(len(steps)):
        # Load image and add to frame list
        path = f"{path2saveResults}/img_{i}.png"
        path2 = f"{path2saveResults}/img_{i}.pdf"
        image = imageio.v2.imread(path)
        frames.append(image)
        # Delete image to avoid clutter in results folder
        os.remove(path)
        os.remove(path2)

    # Create GIF from array of frames
    imageio.mimsave(f"{path2saveResults}/training.gif",  # output path
                    frames,  # array of input frames
                    format="GIF",
                    duration=100)  # optional: duration of frames in ms

