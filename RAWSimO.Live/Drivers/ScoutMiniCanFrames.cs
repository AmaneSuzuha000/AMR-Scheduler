using System;
using System.Buffers.Binary;

namespace RAWSimO.Live.Drivers;

/// <summary>
/// Decodes the AgileX CAN 2.0B (500 kbps) frames that carry SCOUT Mini odometry.
///
/// The official ugv_sdk parses the 0x311 frame but never stores it:
/// agilex_msg_parser_v2.c fills msg->body.odometry_msg, while agilex_base.hpp's ParseCANFrame
/// only dispatches to the core/actuator/common-sensor updaters, none of which handle
/// AgxMsgOdometry. The ROS 2 wrapper therefore publishes dead-reckoned velocity
/// (scout_messenger.hpp integrates the 0x221 motion-state speeds), not wheel odometry.
/// This class decodes 0x311 directly so the vehicle's own wheel counters can drive the pose.
///
/// Frame layout (agilex_protocol_v2.h): OdometryFrame { struct32_t left_wheel; struct32_t right_wheel; }
/// with struct32_t { msb, high_byte, low_byte, lsb }, i.e. big-endian on the wire.
/// The SDK reassembles them as lsb | low&lt;&lt;8 | high&lt;&lt;16 | msb&lt;&lt;24, which is exactly a
/// big-endian 32-bit read. Both wheels are signed cumulative counters.
/// </summary>
public static class ScoutMiniCanFrames
{
    /// <summary>CAN identifier of the odometer feedback frame.</summary>
    public const uint OdometryFrameId = 0x311;

    /// <summary>CAN identifier of the motion-state feedback frame (linear/angular speed).</summary>
    public const uint MotionStateFrameId = 0x221;

    /// <summary>CAN identifier of the system-state frame (mode, battery, faults).</summary>
    public const uint SystemStateFrameId = 0x211;

    /// <summary>CAN identifier of the version-response frame.</summary>
    public const uint VersionResponseFrameId = 0x4A1;

    /// <summary>SCOUT Mini is specified to talk CAN 2.0B at this bitrate.</summary>
    public const int BaudRate = 500_000;

    /// <summary>Reads the two wheel counters out of an 8-byte 0x311 payload.</summary>
    public static WheelCounters DecodeOdometry(ReadOnlySpan<byte> data)
    {
        if (data.Length < 8)
            throw new ArgumentException("odometry frame must carry 8 bytes", nameof(data));

        // The SDK's byte-order reassembly is equivalent to a big-endian 32-bit read.
        var left = BinaryPrimitives.ReadInt32BigEndian(data.Slice(0, 4));
        var right = BinaryPrimitives.ReadInt32BigEndian(data.Slice(4, 4));
        return new WheelCounters(left, right);
    }

    /// <summary>Reads the 0x211 system-state payload: mode, battery voltage, failure bits.</summary>
    /// <remarks>
    /// Table 3.1 of the SCOUT MINI manual (V2.0.1) defines the layout. CAN 2.0B here uses the
    /// Motorola (big-endian) byte order, so the voltage is a big-endian uint16.
    ///
    /// This frame is the authoritative source for the quantities the AgileX-customised RC
    /// transmitter shows on its screen (Vol / Car / Batter / Fault Code). Whatever the 2.4 GHz
    /// link carries, it can carry no more than what this frame provides; therefore decoding
    /// 0x211 gives us those fields without touching the radio at all.
    /// </remarks>
    public static SystemState DecodeSystemState(ReadOnlySpan<byte> data)
    {
        if (data.Length < 8)
            throw new ArgumentException("system-state frame must carry 8 bytes", nameof(data));

        // byte[2] is the high half and byte[3] the low half -> big-endian uint16.
        // The wire unit is 0.1 V (the manual writes "Actual voltage x 10").
        var rawVoltage = BinaryPrimitives.ReadUInt16BigEndian(data.Slice(2, 2));
        return new SystemState(
            ChassisStatus: data[0],
            ControlMode: data[1],
            BatteryVoltage: rawVoltage / 10.0,
            Failures: data[5],
            Counter: data[7]);
    }

    /// <summary>Reads linear and angular speed out of an 8-byte 0x221 payload.</summary>
    /// <remarks>Both fields are int16 scaled by 1000, per agilex_msg_parser_v2.c.</remarks>
    public static (double Linear, double Angular) DecodeMotionState(ReadOnlySpan<byte> data)
    {
        if (data.Length < 8)
            throw new ArgumentException("motion-state frame must carry 8 bytes", nameof(data));

        var linear = BinaryPrimitives.ReadInt16BigEndian(data.Slice(0, 2)) / 1000.0;
        var angular = BinaryPrimitives.ReadInt16BigEndian(data.Slice(2, 2)) / 1000.0;
        return (linear, angular);
    }
}

/// <summary>Raw cumulative wheel counters from one odometer frame.</summary>
public readonly record struct WheelCounters(int Left, int Right);

/// <summary>Decoded 0x211 system-state frame.</summary>
/// <param name="ChassisStatus">byte[0]: 0x00 normal, 0x02 system error.</param>
/// <param name="ControlMode">byte[1]: 0x00 standby, 0x01 CAN command, 0x03 remote control.</param>
/// <param name="BatteryVoltage">byte[2..3] as volts (wire unit is 0.1 V).</param>
/// <param name="Failures">byte[5] as a raw bit field; see <see cref="SystemFailures"/>.</param>
/// <param name="Counter">byte[7] loop counter, 0-255.</param>
public readonly record struct SystemState(
    byte ChassisStatus,
    byte ControlMode,
    double BatteryVoltage,
    byte Failures,
    byte Counter)
{
    /// <summary>True when byte[0] reports a system error.</summary>
    public bool HasSystemError => ChassisStatus == 0x02;

    /// <summary>Control mode as the manual names it, for logs and evidence tables.</summary>
    public string ControlModeName => ControlMode switch
    {
        0x00 => "standby",
        0x01 => "can_command",
        0x03 => "remote_control",
        _ => $"unknown(0x{ControlMode:X2})",
    };
}

/// <summary>Bit meanings of the 0x211 failure byte (manual Table 3.2).</summary>
[Flags]
public enum SystemFailures : byte
{
    /// <summary>No failure bits set.</summary>
    None = 0,
    /// <summary>bit[0] low-voltage failure; protection trips at 20.5 V.</summary>
    LowVoltageFailure = 1 << 0,
    /// <summary>bit[1] low-voltage warning; buzzer at 22.5 V, chassis still drivable.</summary>
    LowVoltageWarning = 1 << 1,
    /// <summary>bit[2] RC signal lost. The receiver is not hearing the transmitter.</summary>
    RemoteSignalLost = 1 << 2,
    /// <summary>bit[3] drive No.1 communication failure.</summary>
    Drive1Failure = 1 << 3,
    /// <summary>bit[4] drive No.2 communication failure.</summary>
    Drive2Failure = 1 << 4,
    /// <summary>bit[5] drive No.3 communication failure.</summary>
    Drive3Failure = 1 << 5,
    /// <summary>bit[6] drive No.4 communication failure.</summary>
    Drive4Failure = 1 << 6,
}
