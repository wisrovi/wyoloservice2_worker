from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class TimestampMixin(BaseModel):
    """Mixin to add automatic created_at and updated_at timestamps."""
    created_at: Optional[datetime] = Field(None, description="index")
    updated_at: Optional[datetime] = Field(None, description="index")

    def pre_save(self):
        """Hook to update timestamps before saving."""
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now

class SoftDeleteMixin(BaseModel):
    """Mixin to add soft delete capability."""
    deleted_at: Optional[datetime] = Field(None, description="index")

class AuditMixin(TimestampMixin, SoftDeleteMixin):
    """Mixin combining timestamps and soft delete."""
    pass
