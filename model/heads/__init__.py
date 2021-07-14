

__all__ = ['build_head']


def build_head(cfg, htype):
    name = cfg.pop('name')
    if name == 'db_head':
        from .db_head import DBHead
        head = DBHead(**cfg)


    return head



