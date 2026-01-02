"""initial migration"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'request_history',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('endpoint', sa.String(50), nullable=True),
        sa.Column('image_width', sa.Integer(), nullable=True),
        sa.Column('image_height', sa.Integer(), nullable=True),
        sa.Column('predicted_label', sa.String(100), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
    )
    op.create_index('ix_request_history_id', 'request_history', ['id'])

def downgrade():
    op.drop_index('ix_request_history_id')
    op.drop_table('request_history')