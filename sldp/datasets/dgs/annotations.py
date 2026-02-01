from pympi.Elan import Eaf

from sldp.utils.iter import iter_files_in_dir
from sldp.elan.parse import get_linked_videos, get_annotations_per_signer_and_task, check_signers_and_tasks_in_annotations

def get_signer_task_from_tier(tier_id: str, tier_params: dict):
    tier_id = tier_id.strip().lower()
    signer_id = tier_params['PARTICIPANT'].strip().lower()
    task = None
    if tier_id.startswith("lexeme_sign_l"):
        task = 'left_hand'
    elif tier_id.startswith("lexeme_sign_r"):
        task = 'right_hand'
    elif tier_id.startswith("translation_into_english"):
        task = 'translation'
    return signer_id, task

if __name__ == '__main__':
    elan_root = "E:/datasets/sign-language/dgs-corpus/annotations/eaf"
    for file in iter_files_in_dir(elan_root, extensions=('eaf',)):
        eaf = Eaf(file)
        # videos = get_linked_videos(eaf)
        # print(videos)
        # annots = eaf.get_annotation_data_for_tier('Lexeme_Sign_r_A')
        annots = get_annotations_per_signer_and_task(eaf, signer_task_from_tier=get_signer_task_from_tier, verbose=False)
        check_signers_and_tasks_in_annotations(annots)
        break
