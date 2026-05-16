from utils import visualize
from utils.visualize import *
visualize.setup_figs()
fig1_model_preference_issue()
fig2_visualise_class_prob()
fig3_visualise_preference_issue()
fig4_5_s1_cbs_bs_box_plot('imagenet_place')
fig4_5_s1_cbs_bs_box_plot('int_lvis')
visualize.setup_figs(length_width_rate=0.6)
fig6_visualise_grad_balancing()
visualize.setup_figs(length_width_rate=1)
fig4_5_s1_cbs_bs_box_plot('cifar')
visualize.setup_figs()
fig_s2_check_probs_target_class()
