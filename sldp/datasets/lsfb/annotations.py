import re

from sldp.utils.iter import iter_files_in_dir
from sldp.elan.parse import get_linked_videos, get_annotations_per_signer_and_task, check_signers_and_tasks_in_annotations

from pympi.Elan import Eaf


def get_signer_id_from_video_name(video_name: str):
    signer_match = re.search(pattern=r'S(\d){2,4}', string=video_name)
    if signer_match is None:
        return None
    signer_id = int(signer_match.group().replace('S', ''))
    signer_id = f"S{signer_id:0>3}"
    return signer_id


def get_signer_task_from_tier(tier_id: str, tier_params: dict):
    tier_id = tier_id.replace(' ', '').upper()
    signer_match = re.search(pattern=r'S?(\d){2,4}', string=tier_id)
    if signer_match is None:
        return None, None
    signer_id = int(signer_match.group().replace('S', ''))
    signer_id = f'S{signer_id:0>3}'
    task = None
    if re.search(pattern="(MG)|(MAINGAUCHE)|(GAUCHE)", string=tier_id):
        task = 'left_hand'
    elif re.search(pattern="(MD)|(MAINDROITE)|(DROITE)", string=tier_id):
        task = 'right_hand'
    elif re.search(pattern="(TRADUCTION)|(TRADCUTION)", string=tier_id):
        task = 'traduction'
    return signer_id, task


if __name__ == '__main__':
    elan_root = "D:/data/sign-languages/lsfb-cont/annotations/ELAN"
    for file in iter_files_in_dir(elan_root, extensions=('eaf',)):
        eaf = Eaf(file)
        signer_ids = set([get_signer_id_from_video_name(v) for v in get_linked_videos(eaf)]) - {None}
        annots = get_annotations_per_signer_and_task(eaf, get_signer_task_from_tier, verbose=False)
        issues = check_signers_and_tasks_in_annotations(
            annotations_per_signer_and_task=annots,
            only_signers=signer_ids,
            only_tasks={'left_hand', 'right_hand', 'traduction'},
            mandatory_signers=signer_ids,
            mandatory_tasks={'left_hand', 'right_hand'},
        )
        if len(issues) > 0:
            print(file)
            print(issues)
