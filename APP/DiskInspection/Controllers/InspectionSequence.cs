using System;
using System.Threading;
using System.Threading.Tasks;

namespace DiskInspection.Controllers
{
    internal enum InspectionLight { White, Uv }

    /// <summary>Owns the lighting tasks for one cycle, including failure and cancellation.</summary>
    internal sealed class InspectionSequence
    {
        private readonly Func<InspectionLight, bool, Task<bool>> _controlLight;
        private readonly Func<int, CancellationToken, Task> _delay;

        public InspectionSequence(Func<InspectionLight, bool, Task<bool>> controlLight,
            Func<int, CancellationToken, Task> delay = null)
        {
            _controlLight = controlLight;
            _delay = delay ?? Task.Delay;
        }

        public async Task RunAsync(Func<CancellationToken, Task> captureWhite,
            Func<CancellationToken, Task> inspectWhite,
            Func<CancellationToken, Task> captureUv,
            Func<CancellationToken, Task> inspectUv,
            int whiteOnDelay, int whiteOffDelay, int uvOnDelay, int uvOffDelay,
            CancellationToken token)
        {
            await RunStageAsync(InspectionLight.White, captureWhite, inspectWhite,
                whiteOnDelay, whiteOffDelay, token).ConfigureAwait(false);
            await RunStageAsync(InspectionLight.Uv, captureUv, inspectUv,
                uvOnDelay, uvOffDelay, token).ConfigureAwait(false);
        }

        private async Task RunStageAsync(InspectionLight light,
            Func<CancellationToken, Task> capture, Func<CancellationToken, Task> inspect,
            int onDelay, int offDelay, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            bool lightOffConfirmed = false;
            try
            {
                if (!await _controlLight(light, true).ConfigureAwait(false))
                    throw new InvalidOperationException($"Cannot turn on {light} light.");
                await _delay(onDelay, token).ConfigureAwait(false);
                await capture(token).ConfigureAwait(false);

                // Keep the specified light hold time overlapped with image processing.
                // WhenAll observes BOTH tasks even when either fails or is cancelled.
                var inspection = InvokeAsync(inspect, token);
                var switchOff = SwitchOffAsync();
                await Task.WhenAll(inspection, switchOff).ConfigureAwait(false);
                token.ThrowIfCancellationRequested();
            }
            finally
            {
                // Cleanup is deliberately independent of the cancelled cycle token.
                if (!lightOffConfirmed && !await _controlLight(light, false).ConfigureAwait(false))
                    throw new InvalidOperationException($"Cannot confirm {light} light is off.");
            }

            async Task SwitchOffAsync()
            {
                await _delay(offDelay, token).ConfigureAwait(false);
                lightOffConfirmed = await _controlLight(light, false).ConfigureAwait(false);
                if (!lightOffConfirmed)
                    throw new InvalidOperationException($"Cannot turn off {light} light.");
            }
        }

        private static async Task InvokeAsync(Func<CancellationToken, Task> operation, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            await operation(token).ConfigureAwait(false);
        }
    }
}
