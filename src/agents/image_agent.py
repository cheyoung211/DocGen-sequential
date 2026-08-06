"""Image asset generation without LaTeX layout decisions.

The ImageAgent creates pixels and records metadata.  Figure environments are
emitted only by ``LatexIntegratorAgent`` according to the DocumentBlueprint.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.common.state import DocumentGraph, NodeStatus, NodeType


IMAGE_STRATEGY_PROMPT = """
You are a visual prompt engineer for professional technical documents. Convert
the requested concept into a precise image-generation prompt. Create an image
that supports the named document section; do not make any LaTeX or placement
decisions.

Return only this JSON object:
{
  "strategy": "GENERIC",
  "image_prompt": "A concise, professional image-generation prompt with no text in the image"
}
"""

ASSET_OUTPUT_DIR = "outputs/generations"


class ImageAgent:
    """Generate and register visual assets for later LaTeX composition."""

    def __init__(self, llm_client: Any, image_generator: Any) -> None:
        self.llm = llm_client
        self.image_generator = image_generator

    def process_figure_node(
        self,
        node_id: str,
        graph: DocumentGraph,
        request_id: str,
        output_dir: str = ASSET_OUTPUT_DIR,
    ) -> Optional[Dict[str, str]]:
        node = graph.nodes.get(node_id)
        if not node or node.type != NodeType.FIGURE:
            return None

        placement = None
        if graph.blueprint:
            placement = graph.blueprint.figure_slots.get(node_id)
            if placement is None:
                raise ValueError(
                    f"Figure '{node_id}' has no reserved slot in the DocumentBlueprint."
                )

        graph.update_node_status(node_id, NodeStatus.RUNNING)
        assets_dir = os.path.join(output_dir, request_id, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        layout_context = {}
        if placement:
            layout_context = {
                "owner_section": placement.owner_section,
                "layout": placement.layout.value,
                "caption": placement.caption or node.title,
            }
        user_context = (
            f"Figure ID: {node.id}\n"
            f"Caption: {node.title}\n"
            f"Content requirements: {json.dumps(node.spec)}\n"
            f"Document context: {json.dumps(layout_context)}"
        )

        try:
            raw_response = self.llm.generate(
                system_prompt=IMAGE_STRATEGY_PROMPT,
                user_prompt=user_context,
                temperature=0.2,
            )
            strategy = json.loads(self.llm.extract_json_block(raw_response))
            image_prompt = strategy.get("image_prompt")
            if not isinstance(image_prompt, str) or not image_prompt.strip():
                raise ValueError("Image strategy response does not contain image_prompt.")

            image = self.image_generator.generate(prompt=image_prompt)
            asset_filename = f"{node_id}.png"
            output_path = os.path.join(assets_dir, asset_filename)
            image.save(output_path)

            # LaTeX files refer to this path relative to <output_dir>/<request_id>.
            relative_path = f"assets/{asset_filename}"
            registry_entry = graph.register_figure_asset(
                node_id,
                relative_path,
                caption=(placement.caption if placement and placement.caption else node.title),
            )
            if relative_path not in node.assets:
                node.assets.append(relative_path)

            metadata = {
                "asset_id": node_id,
                "path": relative_path,
                "caption": registry_entry.caption or node.title,
                "strategy": strategy.get("strategy", "GENERIC"),
                "prompt": image_prompt,
            }
            graph.update_node_status(
                node_id,
                NodeStatus.DRAFTED,
                content=json.dumps(metadata, ensure_ascii=False),
            )
            print(f"[ImageAgent] Registered asset: {relative_path}")
            return metadata
        except Exception as exc:
            graph.update_node_status(
                node_id,
                NodeStatus.ERROR,
                error=str(exc),
            )
            print(f"[ImageAgent] Image generation failed for {node_id}: {exc}")
            return None
