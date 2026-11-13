"""Projection receipt expression shared by SQL acknowledgement and resource erasure."""

# JSONB text retains JSON quoting when assembling the portable receipt.
PROJECTION_RECEIPT_SQL = """CASE
    WHEN jsonb_typeof(payload->'task_owner_spaces') = 'array' THEN
        ('{"task_owner_spaces":' || (payload->'task_owner_spaces')::text || '}')::jsonb
    ELSE coalesce(
        ('{"filters":{"owner_space":' ||
         (CASE WHEN jsonb_typeof(payload->'filters') = 'object'
               THEN payload #> '{filters,owner_space}'
               ELSE payload #> '{records,0,filters,owner_space}' END)::text || '}}')::jsonb,
        '{}'::jsonb)
    END"""
