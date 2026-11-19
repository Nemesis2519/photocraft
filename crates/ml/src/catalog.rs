//! Reviewed FP32 exports. URLs contain immutable revisions; sizes and SHA-256s come from their
//! Hugging Face LFS metadata. Updating a model is a code review, never a remote manifest update.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelId {
    #[serde(rename = "birefnet-hr-matting")]
    BiRefNet,
    #[serde(rename = "sam2.1-large")]
    Sam2,
    #[serde(rename = "isnet-general-use")]
    IsNet,
    #[serde(rename = "birefnet-lite")]
    BiRefNetLite,
    #[serde(rename = "u2net")]
    U2Net,
    #[serde(rename = "vitmatte-small")]
    ViTMatte,
}

impl ModelId {
    pub fn parse(id: &str) -> Option<Self> {
        match id {
            "birefnet-hr-matting" => Some(Self::BiRefNet),
            "sam2.1-large" => Some(Self::Sam2),
            "isnet-general-use" => Some(Self::IsNet),
            "birefnet-lite" => Some(Self::BiRefNetLite),
            "u2net" => Some(Self::U2Net),
            "vitmatte-small" => Some(Self::ViTMatte),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::BiRefNet => "birefnet-hr-matting",
            Self::Sam2 => "sam2.1-large",
            Self::IsNet => "isnet-general-use",
            Self::BiRefNetLite => "birefnet-lite",
            Self::U2Net => "u2net",
            Self::ViTMatte => "vitmatte-small",
        }
    }
    pub fn index(self) -> usize {
        match self {
            Self::BiRefNet => 0,
            Self::Sam2 => 1,
            Self::IsNet => 2,
            Self::BiRefNetLite => 3,
            Self::U2Net => 4,
            Self::ViTMatte => 5,
        }
    }
    pub fn info(self) -> &'static ModelInfo {
        match self {
            Self::BiRefNet => &MODELS[0],
            Self::Sam2 => &MODELS[1],
            Self::IsNet => &MODELS[2],
            Self::BiRefNetLite => &MODELS[3],
            Self::U2Net => &MODELS[4],
            Self::ViTMatte => &MODELS[5],
        }
    }
}

/// Input scaling a model was trained with, applied to sRGB in 0..=1.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Scaling {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

const IMAGENET: Scaling = Scaling { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
/// Scales 0..=1 to -1..=1.
const HALF: Scaling = Scaling { mean: [0.5; 3], std: [0.5; 3] };

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    pub file: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: ModelId,
    pub label: &'static str,
    pub purpose: &'static str,
    pub input_side: usize,
    pub scaling: Scaling,
    pub revision: &'static str,
    pub license: &'static str,
    pub upstream: &'static str,
    pub export_source: &'static str,
    pub artifacts: &'static [Artifact],
}

impl ModelInfo {
    pub fn download_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.bytes).sum()
    }
}

pub const MODELS: [ModelInfo; 6] = [
    ModelInfo {
        id: ModelId::BiRefNet,
        label: "BiRefNet HR Matting",
        purpose: "Select Subject and Remove Background; detailed, soft edges",
        input_side: 2048,
        scaling: IMAGENET,
        revision: "792518b9d4dfe9793891712677de25e554f948c6",
        license: "MIT",
        upstream: "https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting",
        export_source: "https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx",
        artifacts: &[Artifact {
            file: "model.onnx",
            url: "https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx/resolve/792518b9d4dfe9793891712677de25e554f948c6/model.onnx",
            bytes: 932_150_975,
            sha256: "a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd",
        }],
    },
    ModelInfo {
        id: ModelId::Sam2,
        label: "SAM 2.1 Large",
        purpose: "Object Selection; drag a box around the object",
        input_side: 1024,
        scaling: IMAGENET,
        revision: "3c23431f721e69cae82dbfd0c28fd692cc714021",
        license: "Apache-2.0",
        upstream: "https://huggingface.co/facebook/sam2.1-hiera-large",
        export_source: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX",
        artifacts: &[
            Artifact {
                file: "vision_encoder.onnx",
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/3c23431f721e69cae82dbfd0c28fd692cc714021/onnx/vision_encoder.onnx",
                bytes: 1_350_487,
                sha256: "ad3a8161c6bd6d8beea811300c60e20d626035feaa3c19622ace27b22e25bec8",
            },
            Artifact {
                file: "vision_encoder.onnx_data",
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/3c23431f721e69cae82dbfd0c28fd692cc714021/onnx/vision_encoder.onnx_data",
                bytes: 888_587_456,
                sha256: "88c501a65b6aaeac99b65a275f3f559727404283930d107d499fda8714ba7e5a",
            },
            Artifact {
                file: "prompt_encoder_mask_decoder.onnx",
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/3c23431f721e69cae82dbfd0c28fd692cc714021/onnx/prompt_encoder_mask_decoder.onnx",
                bytes: 213_114,
                sha256: "a21b3b09f6e5f5838d630d92a150914ca58f8f58b603b8f53fc32b03da982203",
            },
            Artifact {
                file: "prompt_encoder_mask_decoder.onnx_data",
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/3c23431f721e69cae82dbfd0c28fd692cc714021/onnx/prompt_encoder_mask_decoder.onnx_data",
                bytes: 20_958_208,
                sha256: "bfdcd4f7ee04298a82fe1aaf1bd7b8edf544d36901e1a886f01892fb915bd63c",
            },
        ],
    },
    ModelInfo {
        id: ModelId::IsNet,
        label: "ISNet",
        purpose: "Select Subject and Remove Background; fast enough for integrated GPUs",
        input_side: 1024,
        scaling: Scaling { mean: [0.5; 3], std: [1.0; 3] },
        revision: "v0.0.0",
        license: "Apache-2.0",
        upstream: "https://github.com/xuebinqin/DIS",
        export_source: "https://github.com/danielgatis/rembg",
        artifacts: &[Artifact {
            file: "model.onnx",
            url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
            bytes: 178_648_008,
            sha256: "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        }],
    },
    ModelInfo {
        id: ModelId::BiRefNetLite,
        label: "BiRefNet Lite",
        purpose: "Select Subject and Remove Background; the most accurate quick model",
        input_side: 640,
        scaling: IMAGENET,
        revision: "173d635935b93839608b9b9039da8d1d212471e9",
        license: "MIT",
        upstream: "https://github.com/ZhengPeng7/BiRefNet",
        export_source: "https://huggingface.co/senty-au/BiRefNet_lite-ONNX-dynamic",
        artifacts: &[Artifact {
            file: "model.onnx",
            url: "https://huggingface.co/senty-au/BiRefNet_lite-ONNX-dynamic/resolve/173d635935b93839608b9b9039da8d1d212471e9/onnx/model.onnx",
            bytes: 180_839_545,
            sha256: "1e0da42f0fde010e32e938bad388457ecefe35806fde9d923421997861ae9391",
        }],
    },
    ModelInfo {
        id: ModelId::U2Net,
        label: "U2-Net",
        purpose: "Finds subjects the other models miss; recommended with either of them",
        input_side: 320,
        scaling: IMAGENET,
        revision: "v0.0.0",
        license: "Apache-2.0",
        upstream: "https://github.com/xuebinqin/U-2-Net",
        export_source: "https://github.com/danielgatis/rembg",
        artifacts: &[Artifact {
            file: "model.onnx",
            url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            bytes: 175_997_641,
            sha256: "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        }],
    },
    ModelInfo {
        id: ModelId::ViTMatte,
        label: "ViTMatte Small",
        purpose: "Sharper hair and fur edges for Select Subject and Remove Background; adds a few seconds",
        // The tile the edge band is solved in; inputs are the picture and a trimap channel.
        input_side: 384,
        scaling: HALF,
        revision: "6bc1297f6140f055a227b6d2cfe8c093281f35d2",
        // Weights under Apache-2.0, trained on Composition-1k, which is licensed for research.
        license: "Apache-2.0",
        upstream: "https://huggingface.co/hustvl/vitmatte-small-composition-1k",
        export_source: "https://huggingface.co/Xenova/vitmatte-small-composition-1k",
        artifacts: &[Artifact {
            file: "model.onnx",
            url: "https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/6bc1297f6140f055a227b6d2cfe8c093281f35d2/onnx/model.onnx",
            bytes: 103_885_865,
            sha256: "bf28d2e0be2c073286e88d60ad649d7123da2749a2d99133fd1098d5887e0225",
        }],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_is_pinned_and_ids_cannot_escape_storage() {
        for m in MODELS {
            assert_eq!(ModelId::parse(m.id.name()), Some(m.id));
            for a in m.artifacts {
                assert!(a.url.starts_with("https://huggingface.co/") || a.url.starts_with("https://github.com/danielgatis/rembg/releases/download/"));
                assert!(a.url.contains(m.revision));
                assert!(a.bytes > 0 && a.bytes < 2_000_000_000);
                assert_eq!(a.sha256.len(), 64);
                assert!(a.sha256.bytes().all(|c| c.is_ascii_hexdigit()));
                assert!(!a.file.contains(['/', '\\']));
            }
        }
        for id in ["../model", "sam3", "", "https://example.com/model.onnx"] {
            assert_eq!(ModelId::parse(id), None);
        }
    }
}
