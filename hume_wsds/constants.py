ShardMapping = {
    # segmentation types (mvad) 
    "v4-vad_ws_mvad.raw.vad.npy": ["v4-vad_ws_mvad", "raw.vad.npy"],
    "v4-vad_ws_mvad.eq.vad.npy":  ["v4-vad_ws_mvad", "eq.vad.npy"],
    "v4-vad_ws_mvad.max.vad.npy": ["v4-vad_ws_mvad", "max.vad.npy"],

    "v4-vad_ws_continuous_mvad.raw.vad.npy": ["v4-vad_ws_continuous_mvad", "raw.vad.npy"],
    "v4-vad_ws_continuous_mvad.eq.vad.npy":  ["v4-vad_ws_continuous_mvad", "eq.vad.npy"],
    "v4-vad_ws_continuous_mvad.max.vad.npy": ["v4-vad_ws_continuous_mvad", "max.vad.npy"],

    "v3-vad_ws_mvad.raw.vad.npy": ["v3-vad_ws_mvad", "raw.vad.npy"],
    "v3-vad_ws_mvad.eq.vad.npy":  ["v3-vad_ws_mvad", "eq.vad.npy"],
    "v3-vad_ws_mvad.max.vad.npy": ["v3-vad_ws_mvad", "max.vad.npy"],


    "v5-diarized_continuous_mvad.diarized.vad.npy": ["v5-diarized_continuous_mvad", "diarized.vad.npy"],
    "v5-diarized_continuous_mvad.diarized.pause_dur.npy": ["v5-diarized_continuous_mvad", "diarized.pause_dur.npy"],
    "v5-diarized_continuous_mvad.diarized.pause_energy.npy": ["v5-diarized_continuous_mvad", "diarized.pause_energy.npy"],
    "v5-diarized_continuous_mvad.diarized.speaker.npy": ["v5-diarized_continuous_mvad", "diarized.speaker.npy"],

    "v5-diarized_mvad.diarized.vad.npy": ["v5-diarized_mvad", "diarized.vad.npy"],
    "v5-diarized_mvad.diarized.pause_dur.npy": ["v5-diarized_mvad", "diarized.pause_dur.npy"],
    "v5-diarized_mvad.diarized.pause_energy.npy": ["v5-diarized_mvad", "diarized.pause_energy.npy"],
    "v5-diarized_mvad.diarized.speaker.npy": ["v5-diarized_mvad", "diarized.speaker.npy"],

    
    # audio quality 
    "c50": ["acoustic_scores_raw", "c50"],
    "dbu": ["acoustic_scores_raw", "dbu"],
    "snr": ["acoustic_scores_raw", "snr"],

    "c50": ["acoustic_scores_continuous", "c50"],
    "dbu": ["acoustic_scores_continuous", "dbu"],
    "snr": ["acoustic_scores_continuous", "snr"],

    
    "pq": ["pquality_wmusic_scores_raw", "pq"],
    "music_qual": ["pquality_wmusic_scores_raw", "music_qual"],
    "pq": ["pquality_scores_continuous", "pq"],

    
    # tokens 
    "dtok_25_10_10_vocab_256_32k.dtok_global.npy": ["dtok_25_10_10_vocab_256_32k", "dtok_global.npy"],
    "dtok_25_10_10_vocab_256_32k.dtok_level_1.npy": ["dtok_25_10_10_vocab_256_32k", "dtok_level_1.npy"],
    "dtok_25_10_10_vocab_256_32k.dtok_level_2.npy": ["dtok_25_10_10_vocab_256_32k", "dtok_level_2.npy"],
    "dtok_25_10_10_vocab_256_32k.dtok_level_3.npy": ["dtok_25_10_10_vocab_256_32k", "dtok_level_3.npy"],
    
    "dtok_level_1_16k.npy": ["dtok_v2_ml_50hz_32x16384_graphemes_key16k", "dtok_level_1_16k.npy"],
    "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_global.npy": ["dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder", "dtok_global.npy"],
    "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_level_1.npy": ["dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder", "dtok_level_1.npy"],
    "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_level_2.npy": ["dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder", "dtok_level_2.npy"],
    "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.source_start_end_time.npy": ["dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder", "source_start_end_time.npy"],
    "dtok_v2_ml_50hz_32x16384_graphemes_key16k.source_start_end_time.npy": ["dtok_v2_ml_50hz_32x16384_graphemes_key16k", "source_start_end_time.npy"],
    
    # transcripts
    "nemo_transcription.txt": ["transcription_parakeet-ctc-1-1b_raw", "nemo_transcription.txt"],
    "transcription_ws_raw.txt": ["transcription_ws_raw", "txt"],
    "transcription_wslang_raw.txt": ["transcription_wslang_raw", "txt"],
    "transcription_wslang_continuous.txt": ["transcription_wslang_continuous", "txt"],


    # "__key__": ["stoks_raw", "__key__"],
    # "aes": ["meta_aes_quality_scores_raw", "aes"],
    # "atoks.npy": ["atoks_raw", "atoks.npy"],
    # "audio_quality": ["mos_scores_utmos_raw", "audio_quality"],
    # "dac_32c.npy": ["dac_32c", "dac_32c.npy"],
    # "dtok_25hz_vocab_512x3_global_4p4_v2.dtok_global.npy": ["dtok_25hz_vocab_512x3_global_4p4_v2", "dtok_global.npy"],
    # "dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_1.npy": ["dtok_25hz_vocab_512x3_global_4p4_v2", "dtok_level_1.npy"],
    # "dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_2.npy": ["dtok_25hz_vocab_512x3_global_4p4_v2", "dtok_level_2.npy"],
    # "dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_3.npy": ["dtok_25hz_vocab_512x3_global_4p4_v2", "dtok_level_3.npy"],
    # "dtok_prosmimic_48.npy": ["dtok_prosmimic_48", "dtok_prosmimic_48.npy"],
    # "duration": ["utmos_scores_raw", "duration"],
    # "emphassess": ["emphasis_raw", "emphassess"],
    # "loudness": ["clipping_events", "loudness"],
    # "mean_48.npy": ["dtok_prosmimic_48", "mean_48.npy"],
    # "music_qual": ["pquality_wmusic_scores_raw", "music_qual"],
    # "n_clip_events": ["clipping_events", "n_clip_events"],
    # "n_speakers": ["speaker_activity_raw", "n_speakers"],
    # "natmos": ["natmos_v0", "natmos"],
    # "pitch_curve.npy": ["rmvpe_pitch_curve_raw", "pitch_curve.npy"],
    # "pquality_scores_raw.pq": ["pquality_scores_raw", "pq"],
    # "sf_mean": ["spectral_flatness", "sf_mean"],
    # "sf_time": ["spectral_flatness", "sf_time"],
    # "sf_values": ["spectral_flatness", "sf_values"],
    # "speaker_activity": ["speaker_activity_raw", "speaker_activity"],
    # "spk_emb.npy": ["stoks_raw", "spk_emb.npy"],
    # "stoks.npy": ["stoks_raw", "stoks.npy"],
    # "utmos": ["utmos_scores_raw", "utmos"],
    # "vocmimic": ["vocmimic_raw", "vocmimic"],
    # "voxsim.npy": ["voxsim", "voxsim.npy"],
    # "word_alignment_whisperx_raw.txt": ["word_alignment_whisperx_raw", "txt"],
    # "zcr": ["clipping_events", "zcr"],
}