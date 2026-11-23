using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Tracking;

/// <summary>Locks scene/run provenance for one network recording or TCP connection.</summary>
public sealed class FleetRunIdentity
{
    private string? _run;
    private string? _scene;

    /// <summary>Legacy read-only hellos may establish identity in their first execution frame.</summary>
    public FleetRunIdentity(FleetControlState? control = null)
    {
        if (control is not null) Accept(control.SceneId, control.RunId);
    }

    /// <summary>Validate before publishing an observation, hello or control receipt to consumers.</summary>
    public void Observe(LiveMessage message)
    {
        var control = message switch
        {
            FleetHelloMessage hello => hello.Control,
            FleetFrameMessage frame => frame.Control,
            FleetControlResultMessage receipt => receipt.Control,
            _ => null
        };
        if (control is not null) Accept(control.SceneId, control.RunId);
        if (message is FleetFrameMessage { ExecutionIdentity: { } execution })
            Accept(execution.SceneId, execution.RunId);
    }

    private void Accept(string scene, string run)
    {
        _run ??= run;
        _scene ??= scene;
        if (_run != run || _scene != scene)
            throw new FormatException("Mixed scene/run state; a different run requires a new connection or recording.");
    }
}
