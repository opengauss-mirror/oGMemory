CREATE OR REPLACE FUNCTION ogmemory_space_source_changed() RETURNS trigger AS $$
DECLARE
    node_uri text;
    tenant text;
    revision integer;
    change_id text;
BEGIN
    IF TG_OP = 'DELETE' THEN
        node_uri := OLD.uri;
        tenant := OLD.account_id;
        revision := OLD.version;
    ELSE
        node_uri := NEW.uri;
        tenant := NEW.account_id;
        revision := NEW.version;
        IF TG_OP = 'UPDATE' AND
           (NEW.uri, NEW.status, NEW.content, NEW.abstract, NEW.overview, NEW.version) IS NOT DISTINCT FROM
           (OLD.uri, OLD.status, OLD.content, OLD.abstract, OLD.overview, OLD.version) THEN
            RETURN NEW;
        END IF;
    END IF;
    IF node_uri ~ '^ctx://[^/]+/(users|agents)/[^/]+/memories/' THEN
        change_id := md5(random()::text || clock_timestamp()::text || node_uri);
        INSERT INTO outbox_events(event_id,event_type,uri,account_id,payload,status)
        VALUES(change_id,'SOURCE_CHANGED',node_uri,tenant,
               json_build_object('change_id',change_id,'source_uri',node_uri,
                                  'content_rev',revision,'change_type',TG_OP)::jsonb,
               'SOURCE_PENDING');
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.uri <> NEW.uri
       AND OLD.uri ~ '^ctx://[^/]+/(users|agents)/[^/]+/memories/' THEN
        change_id := md5(random()::text || clock_timestamp()::text || OLD.uri);
        INSERT INTO outbox_events(event_id,event_type,uri,account_id,payload,status)
        VALUES(change_id,'SOURCE_CHANGED',OLD.uri,OLD.account_id,
               json_build_object('change_id',change_id,'source_uri',OLD.uri,
                                  'content_rev',OLD.version,'change_type','MOVE')::jsonb,
               'SOURCE_PENDING');
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ogmemory_space_source_change ON context_nodes;
CREATE TRIGGER ogmemory_space_source_change
AFTER INSERT OR UPDATE OR DELETE ON context_nodes
FOR EACH ROW EXECUTE PROCEDURE ogmemory_space_source_changed();

CREATE INDEX IF NOT EXISTS idx_oe_space_dispatch_account
ON outbox_events (account_id,seq) WHERE event_type='SOURCE_CHANGED' AND status='SOURCE_PENDING';
