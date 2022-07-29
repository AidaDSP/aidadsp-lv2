#include <lsp-plug.in/dsp/dsp.h>
#include <lsp-plug.in/dsp-units/units.h>
#include <lsp-plug.in/dsp-units/sampling/Sample.h>
#include <lsp-plug.in/dsp-units/filters/Filter.h>
#include <string.h>
#include <stdio.h>

static int process_file(const char *in, const char *out)
{
    // We will process file as a single audio sample
    // That can take many memory for large files
    // but for demo it's OK
    lsp::dspu::Sample s;
    lsp::dspu::Filter f;
    lsp::dspu::filter_params_t fp;
    lsp::io::Path path;

    // We consider pathnames encoded using native system encoding.
    // That's why we use io::Path::set_native() method
    path.set_native(in);
    if (s.load(&path) != lsp::STATUS_OK)
    {
        fprintf(stderr, "Error loading audio sample from file: %s\n", in);
        return -1;
    }

    // Apply +6 dB hi-shelf filter at 1 kHz
    fp.nType        = lsp::dspu::FLT_BT_BWC_HISHELF;
    fp.fFreq        = 1000.0f;
    fp.fFreq2       = 1000.0f;
    fp.fGain        = lsp::dspu::db_to_gain(6.0f);
    fp.nSlope       = 2;
    fp.fQuality     = 0.0f;

    f.init(NULL);   // Use own internal filter bank
    f.update(s.sample_rate(), &fp); // Apply filter settings

    // Now we need to process each channel in the sample
    for (size_t i=0; i<s.channels(); ++i)
    {
        float *c    = s.channel(i);     // Get channel data
        f.clear();                      // Reset internal memory of filter
        f.process(c, c, s.samples());
    }

    // Resample the processed sample to 48 kHz sample rate
    s.resample(48000);

    // Save sample to file
    path.set_native(out);
    if (s.save(&path) < 0)
    {
        fprintf(stderr, "Error saving audio sample to file: %s\n", out);
        return -2;
    }

    return 0;
}

int main(int argc, const char **argv)
{
    lsp::dsp::context_t ctx;

    if (argc < 3)
    {
        fprintf(stderr, "Input file name and output file name required");
        return -1;
    }

    // Initialize DSP
    lsp::dsp::init();
    lsp::dsp::start(&ctx);

    int res = process_file(argv[1], argv[2]);

    lsp::dsp::finish(&ctx);

    return res;
}