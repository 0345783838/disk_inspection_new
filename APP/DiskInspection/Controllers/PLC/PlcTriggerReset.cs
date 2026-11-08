using System;
using System.Threading;
using System.Threading.Tasks;

namespace DiskInspection.Controllers.PLC
{
    internal static class PlcTriggerReset
    {
        internal static async Task RunAsync(Func<bool> reset, CancellationToken token,
            Func<int, CancellationToken, Task> delay = null)
        {
            const int maxAttempts = 3;
            const int retryDelayMs = 200;
            delay = delay ?? Task.Delay;

            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                token.ThrowIfCancellationRequested();
                bool success = reset();
                token.ThrowIfCancellationRequested();
                //if (success) return;

                if (attempt < maxAttempts)
                    await delay(retryDelayMs, token).ConfigureAwait(false);
            }

            //throw new InvalidOperationException("Cannot reset PLC trigger after 3 attempts.");
            return;
        }
    }
}
